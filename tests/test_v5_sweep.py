"""V5 item 3 — the lattice is actually swept.

Every bound segment used to sit at ``x = 0`` whatever ``Wing.sweep_deg`` had
been scored (``vlm.py``), so sweep existed only as a reduced-order correction:
``llt.swept_section_slope`` on the section, a Raymer form factor on the drag
and a Raymer term on the weight. The neutral point never moved.

The trap this file guards is the DOUBLE COUNT. ``objective.py`` hands the VLM
the UNSWEPT ``a_lin/beta`` and applies simple sweep theory only on the
lifting-line path — so the geometry is owned here and the section slope
there. If a future session ever passes ``swept_section_slope`` into a swept
lattice, the sweep is charged twice and
``test_the_lattice_does_not_also_reduce_the_section_slope`` goes red.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo.geometry import Wing
from aerobo.vlm import VLM, TailSurface

W = dict(b=10.0, S=10.0, taper=0.6, twist_tip_deg=-2.0)
RECT = dict(b=10.0, S=10.0, taper=1.0)


def _m(sweep_deg: float, wing_kw=None, **kw) -> VLM:
    return VLM(Wing(**(wing_kw or W), sweep_deg=sweep_deg), N=60, V=30.0, **kw)


def _mac_quarter_chord_x(wing: Wing, sweep_deg: float) -> float:
    """Independently written closed form for where the MAC's c/4 lands.

    ``x_mac = y_mac tan(Lambda)`` with ``y_mac`` the chord-weighted spanwise
    centroid of the semi-span — the standard result, computed here from the
    wing's own chord law rather than from anything ``vlm`` did.
    """
    y = np.linspace(0.0, wing.b / 2.0, 8001)
    c = wing.chord(y)
    y_mac = float(np.trapezoid(c * y, y) / np.trapezoid(c, y))
    return y_mac * float(np.tan(np.deg2rad(sweep_deg)))


# ----------------------------------------------------------- the geometry

def test_zero_sweep_is_bit_for_bit_with_a_winglet_and_a_tail():
    """The GEOMETRY is bit-for-bit; the SOLVE is only float-identical.

    Every published lattice must be reproduced exactly, and that claim lives
    in the panel arrays — they are pure arithmetic on the inputs. The FORCES
    are not a fair place to make it: twelve builds of one identical geometry
    return two distinct values of CL, 5.6e-17 apart (1.8e-16 relative),
    because ``lu_factor`` goes through BLAS and BLAS is free to pick a
    different reduction order run to run. Asserting equality there measures
    the linear algebra's mood, not this module's behaviour — and it duly
    went red on a day when importing a plotting library changed the thread
    pool. So: arrays exact, forces to the measured floor.
    """
    kw = dict(winglet_h_frac=0.12, winglet_cant_deg=80.0,
              tail=TailSurface(S=1.75, x=5.5, z=0.5, AR=4.0, N=20))
    a = VLM(Wing(**W), N=60, V=30.0, **kw)
    b = _m(0.0, **kw)
    assert np.array_equal(a.A3, b.A3) and np.array_equal(a.B3, b.B3)
    assert np.array_equal(a.st3, b.st3)
    assert np.array_equal(a.c, b.c) and np.array_equal(a.twist, b.twist)
    assert np.array_equal(a.width, a.width_wake)      # the wake-trace shortcut
    ra, rb = a.solve(0.07, i_t=0.01), b.solve(0.07, i_t=0.01)
    for got, want in ((ra.CL, rb.CL), (ra.CDi, rb.CDi), (ra.e, rb.e)):
        assert got == pytest.approx(want, rel=1e-14)


def test_the_quarter_chord_line_actually_moves_aft():
    flat, swept = _m(0.0), _m(25.0)
    assert np.array_equal(flat.x, np.zeros_like(flat.x))
    # ``x`` is the bound segment's MIDPOINT and ``y`` a station midpoint, so
    # they are compared through the panel EDGES, which are the same points.
    tanL = np.tan(np.deg2rad(25.0))
    assert swept.A3[:, 0] == pytest.approx(np.abs(swept.A3[:, 1]) * tanL,
                                           rel=1e-12, abs=1e-15)
    assert swept.B3[:, 0] == pytest.approx(np.abs(swept.B3[:, 1]) * tanL,
                                           rel=1e-12, abs=1e-15)
    assert swept.x.min() >= 0.0                  # the root stays the datum


def test_the_panel_station_is_mirror_symmetric():
    """``x`` per panel must be a palindrome on a symmetric swept wing.

    ``A3`` is always the more-negative-y edge of a bound segment, so on the
    STARBOARD side it is the inboard end and on the PORT side the outboard
    one. Taking it as "the panel's station" therefore biases port and
    starboard in opposite directions by half a panel's sweep offset — an
    aeroplane whose two wings sit at different x. The moment arm is the
    segment's midpoint, and this is the property that says so.
    """
    m = _m(25.0)
    wing_p = ~(m.is_winglet | m.is_tail | m.is_vertical)
    x = m.x[wing_p]
    assert x == pytest.approx(x[::-1], rel=1e-12, abs=1e-15)
    # ...and it is not a trivial palindrome: the wing IS swept
    assert x.max() - x.min() > 0.5


def test_a_vertical_tip_device_inherits_its_junction_station():
    """Sweep is a PLANFORM-view angle, so a vertical device does not sweep.

    Its |y| stops changing once it leaves the wing plane, which is exactly
    what a fin standing on a swept tip does. Using the DEVELOPED arc instead
    would rake it aft for no reason.
    """
    m = _m(25.0, winglet_h_frac=0.15, winglet_cant_deg=90.0)
    dev = m.x[m.is_winglet]
    assert dev.max() - dev.min() < 1e-9
    assert dev.max() > 0.5 * m.x.max()           # ...and it IS out at the tip


# -------------------------------------------------------- the consequences

def test_the_neutral_point_moves_aft_by_about_the_mac_offset():
    """The whole reason a swept lattice is worth having.

    Checked against the independently written ``y_mac tan(Lambda)``, which
    the lattice does not compute — the agreement is loose (the tail's own
    contribution does not move) but the SHIFT is the thing.
    """
    tail = TailSurface(S=1.75, x=5.5, z=0.5, AR=4.0, N=20)
    x0 = _m(0.0, tail=tail).neutral_point()
    for L in (10.0, 20.0, 30.0):
        shift = _m(L, tail=tail).neutral_point() - x0
        closed = _mac_quarter_chord_x(Wing(**W), L)
        assert shift > 0.0
        assert 0.8 < shift / closed < 1.2, (
            f"at {L} deg the np moved {shift:.4f}, MAC c/4 is {closed:.4f}")


def test_the_lattice_does_not_also_reduce_the_section_slope():
    """The double-count guard.

    Simple sweep theory would take the lift-curve slope to ``a cos(Lambda)``.
    The lattice reduces it too — through the geometry — but by LESS, because
    a swept lattice is not a strip model. If the two are ever composed the
    ratio falls to (or below) ``cos(Lambda)`` and this goes red.
    """
    def slope(L: float) -> float:
        m, e = _m(L), 1e-5
        return (m.solve(e).CL - m.solve(-e).CL) / (2.0 * e)

    base = slope(0.0)
    for L in (10.0, 20.0, 30.0):
        ratio = slope(L) / base
        cosL = float(np.cos(np.deg2rad(L)))
        assert ratio < 1.0, "sweep must cost lift-curve slope"
        assert ratio > cosL, (
            f"at {L} deg the slope ratio {ratio:.4f} is at or below "
            f"cos(Lambda) {cosL:.4f} — the sweep is being counted twice")
    # ...and the VLM keeps the section slope it was HANDED
    assert _m(30.0).a == pytest.approx(2.0 * np.pi)


def test_induced_drag_is_charged_on_the_wake_trace_not_the_panel():
    """A swept panel's trace is shorter than the panel — Munk collapses x.

    Charging the 3-D bound length would inflate CDi by ``1/cos(Lambda)``:
    +41 % at 45 deg. What sweep actually costs at fixed CL is the loading
    becoming less elliptic, which is a far smaller number and the one the
    lattice must report.
    """
    def cdi(L: float) -> float:
        return _m(L, wing_kw=RECT).solve_trim(0.4)[1].CDi

    # the geometric fact first: a swept panel's wake trace IS its panel
    # length times cos(Lambda)
    m = _m(30.0, wing_kw=RECT)
    wing_p = ~(m.is_winglet | m.is_tail | m.is_vertical)
    assert (m.width_wake[wing_p]
            == pytest.approx(m.width[wing_p] * np.cos(np.deg2rad(30.0)),
                             rel=1e-12))

    base = cdi(0.0)
    for L in (30.0, 45.0):
        rise = cdi(L) / base
        wrong = 1.0 / float(np.cos(np.deg2rad(L)))
        # charging the panel would multiply the loading rise BY ``wrong``;
        # what is left is the loading rise alone, which is well under half of
        # the inflation at both angles (measured 0.47 and 0.34 of it)
        assert 1.0 < rise, "sweep does cost span efficiency"
        assert rise - 1.0 < 0.6 * (wrong - 1.0), (
            f"at {L} deg CDi rose {rise:.4f}x; charging the panel length "
            f"instead of the trace would give about {rise * wrong:.4f}x")


def test_sweep_and_dihedral_compose():
    """Both are geometry and both must survive the other being present."""
    m = VLM(Wing(**W, sweep_deg=20.0, dihedral_deg=5.0), N=60, V=30.0)
    assert m.x.max() > 0.0                       # swept
    assert m.z.max() > 0.0                       # and canted
    # the sweep is measured on the PROJECTED y, which the cant has already
    # shortened — so the two compose in the order a drawing states them
    assert m.A3[:, 0] == pytest.approx(
        np.abs(m.A3[:, 1]) * np.tan(np.deg2rad(20.0)), rel=1e-12, abs=1e-15)
