"""Sweep IS a wing-side dihedral effect, and it is not the cant's lever.

``geometry.Wing.dihedral_deg`` carried the comment "This is the ONLY source
of Cl_beta a lifting surface has", and four shells said the same thing to a
user in as many words. It is false on the package's own lattice: a FLAT,
uncanted wing swept 30 deg reports ``Cl_beta`` -0.0307 at alpha 3 deg, which
is a quarter of what 5 deg of dihedral buys and not zero.

What is true — and what the comment was reaching for — is that the two
sources are different KINDS of number, which is why one is the roll lever and
the other is not:

* a CANT puts a y-component in the panel normal, so a sideslip pushes on it
  through the BOUNDARY CONDITION. The circulation redistributes, and the
  answer does not move with lift: -0.078412 at alpha 1, 3 and 6 deg alike.
* SWEEP leaves a flat wing's normals pointing straight up, so the sideslip
  RHS is identically zero and the circulation does not move at all. The
  rolling moment arrives entirely through the local-velocity
  Kutta-Joukowski term — the crossflow crossed with a bound segment that now
  has an x-extent — and is therefore exactly proportional to lift:
  ``Cl_beta = -0.22 CL tan(Lambda)`` across every alpha measured.

So sweep's dihedral effect is largest where the aeroplane is slowest and
vanishes at cruise, and it comes with a matching rise in ``Cl_r`` that eats
the spiral margin it just bought. That is the measured reason the n=42 paired
study (RESULTS_SESSION67_WING_CANT.md) found the searched sweep row moved
0.19 deg between an L/D arm and a spiral-weighted one (sign p 0.188) while
the dihedral row moved 9.00 deg (sign p 4.4e-07): when the objective pays for
a convergent spiral it buys cant, not sweep.

Every number below is re-derived from the lattice, never pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import dynamics, geometry
from aerobo.vlm import VLM, VerticalSurface

FIN = VerticalSurface(height=1.2, chord=0.8, x=4.0)


def _deck(sweep=0.0, dihedral=0.0, alpha_deg=3.0, fin=False, **kw):
    w = geometry.Wing(b=10.0, S=10.0, taper=0.6, sweep_deg=sweep,
                      dihedral_deg=dihedral)
    m = VLM(w, N=60, vertical=FIN if fin else None)
    return dynamics.deck(m, x_cg=0.0, mac=w.mac,
                         alpha=np.deg2rad(alpha_deg), **kw)


def test_a_flat_swept_wing_rolls_out_of_a_sideslip():
    """The claim the comment denied: sweep alone makes Cl_beta, and it is
    stabilising. The unswept twin is the zero this is measured against."""
    flat = _deck()
    swept = _deck(sweep=30.0)
    assert abs(flat.Cl_beta) < 1e-9, flat.Cl_beta
    assert swept.Cl_beta < -1e-3, swept.Cl_beta


def test_the_swept_dihedral_effect_is_proportional_to_lift():
    """``Cl_beta / (CL tan Lambda)`` is one number over the whole alpha band.

    That is the property that makes sweep a different lever from a cant: the
    roll stiffness a swept wing has at the stall is gone at cruise.
    """
    for sweep in (10.0, 20.0, 30.0):
        ratios = []
        for alpha_deg in (1.0, 3.0, 6.0, 9.0):
            d = _deck(sweep=sweep, alpha_deg=alpha_deg)
            CL = d.CL_alpha * np.deg2rad(alpha_deg)
            ratios.append(d.Cl_beta / (CL * np.tan(np.deg2rad(sweep))))
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-6), (
            sweep, ratios)
        # measured -0.220 (10 deg) to -0.227 (30 deg); the band is wide
        # enough to be a tripwire on the MECHANISM and not on a digit
        assert -0.30 < ratios[0] < -0.15, (sweep, ratios[0])


def test_the_canted_dihedral_effect_is_not():
    """The same sweep of alpha against a cant: the answer does not move."""
    vals = [_deck(dihedral=5.0, alpha_deg=a).Cl_beta
            for a in (1.0, 3.0, 6.0, 9.0)]
    for v in vals[1:]:
        assert v == pytest.approx(vals[0], rel=1e-9), vals
    assert vals[0] < 0.0


def test_sweeps_cl_beta_is_the_local_velocity_force_and_nothing_else():
    """Switch the local-velocity force off and the swept wing's dihedral
    effect is EXACTLY zero, while the cant's is untouched.

    This is the mechanism, asserted where it can be seen: a flat swept wing's
    normals have no y-component, so the sideslip right-hand side ``-u.n``
    vanishes and the circulation never moves.
    """
    assert _deck(sweep=30.0, local_velocity=False).Cl_beta == 0.0
    canted = _deck(dihedral=5.0)
    assert (_deck(dihedral=5.0, local_velocity=False).Cl_beta
            == pytest.approx(canted.Cl_beta, rel=1e-12))


def test_a_degree_of_cant_beats_a_degree_of_sweep_by_an_order():
    """Per degree, at the same lift, on the same wing."""
    per_deg_sweep = abs(_deck(sweep=10.0).Cl_beta) / 10.0
    per_deg_cant = abs(_deck(dihedral=5.0).Cl_beta) / 5.0
    assert per_deg_cant > 10.0 * per_deg_sweep, (per_deg_cant, per_deg_sweep)


def test_sweep_does_not_buy_the_spiral_it_spends_it():
    """The whole point, and the reason the searched sweep row moves nothing.

    Sweep raises ``Cl_r`` — the term the spiral criterion SUBTRACTS — about
    as fast as it raises ``|Cl_beta|``, so on an aeroplane that already has a
    fin and a cant the margin FALLS as the wing is swept back.
    """
    margins, cl_r = [], []
    for sweep in (0.0, 10.0, 20.0, 30.0):
        d = _deck(sweep=sweep, dihedral=5.0, fin=True)
        margins.append(dynamics.spiral_margin_of(d))
        cl_r.append(d.Cl_r)
    assert margins[0] > 0.0, margins          # the cant converged it
    assert margins == sorted(margins, reverse=True), margins
    assert cl_r == sorted(cl_r), cl_r
    assert margins[-1] < 0.2 * margins[0], margins
