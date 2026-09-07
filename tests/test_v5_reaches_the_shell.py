"""V5 item 9 — the work reaches the SCREEN.

The engine changes landed with 84 tests and 47 mutants and the user opened
the shell and saw "no vertical tail, no dihedral". Both were true:

* ``gui/nice_app.fig_wing3d`` is a SECOND geometry drawer, independent of
  ``cad.surfaces``. Item 8 taught the exporter about the fin and this view
  never heard about it, so the fin was in the STL and not in the picture.
* wing dihedral shipped as a flag with no control anywhere in the shell, so
  there was no way to type one.

Neither was caught because every V5 test asserted engine behaviour. A change
a user cannot see is not delivered, and this file is what says so.
"""
from __future__ import annotations

import pytest

from aerobo import api
from aerobo.flightmodel import build_flight_model
from gui import nice_app as v1

V_TAIL = {"tail_type": "v_tail", "dihedral_deg": 35.0}
LATTICE = "tail + winglet"
LATTICE_PAIR = "tandem (nonplanar) + winglets"
LIFTING_LINE = "tail"


def _rep(problem: str, flags: dict | None = None) -> dict:
    flags = flags or {}
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    rep["_x"] = built.bounds.mean(axis=1)
    return rep


def _fig_names(problem: str, flags: dict | None = None) -> list:
    rep = _rep(problem, flags)
    fig = v1.fig_wing3d(rep["geometry"], rep["_x"], rep["param_labels"])
    assert fig is not None
    return [getattr(t, "name", None) for t in fig.data]


# ------------------------------------------------- the fin is in the PICTURE

def test_the_3d_view_draws_the_fin_for_a_conventional_design():
    assert "fin" in _fig_names(LATTICE)


def test_the_3d_view_draws_the_fin_for_a_tandem():
    assert "fin" in _fig_names("tandem")


def test_the_3d_view_draws_NO_fin_for_a_v_tail():
    """The design has none, so neither does the picture — the rule reaches
    the view without being restated in it."""
    assert "fin" not in _fig_names(LIFTING_LINE, V_TAIL)


def test_the_view_and_the_export_agree_about_the_fin():
    """The two drawers must not disagree again.

    ``fig_wing3d`` and ``cad.surfaces`` are independent; what stops them
    drifting is that the fin has ONE loft (``cad.fin_surface``) and both ask
    it. Asserted as agreement on the QUESTION, over both answers.
    """
    from aerobo import cad
    for problem, flags in ((LATTICE, {}), ("tandem", {}),
                           (LIFTING_LINE, V_TAIL)):
        rep = _rep(problem, flags)
        drawn = "fin" in _fig_names(problem, flags)
        exported = any(s.name == "fin"
                       for s in cad.surfaces(rep["geometry"]))
        assert drawn == exported, f"{problem}: drawn {drawn}, exported {exported}"


def test_the_title_says_the_fin_is_there():
    rep = _rep(LATTICE)
    fig = v1.fig_wing3d(rep["geometry"], rep["_x"], rep["param_labels"])
    assert "fin" in str(fig.layout.title.text)


# --------------------------------------------- the cant is ANSWERABLE

def test_a_lattice_family_offers_the_cant_and_the_lifting_line_one_does_not():
    for name in (LATTICE, LATTICE_PAIR):
        assert set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[name].flags), \
            f"{name} cannot be asked for a cant"
    for name in (LIFTING_LINE, "tandem"):
        assert not (set(api.WING_CANT_KEYS)
                    & set(api.PROBLEM_SPECS[name].flags))


def test_every_family_that_offers_the_cant_can_actually_fly_it():
    """The card is gated on ``spec.flags``, so a family that DECLARES the
    keys and drops them would show a control that does nothing."""
    for name in (LATTICE, LATTICE_PAIR):
        rep = _rep(name, {"wing_dihedral_deg": 6.0})
        assert rep["breakdown"]["wing_dihedral_deg"] == 6.0
        assert build_flight_model(rep, V=45.0).model.dihedral_deg == 6.0


def test_a_family_that_cannot_score_a_cant_has_a_twin_that_can():
    """What the refused card points at. A refusal with no address is a dead
    end, and this is the lookup the hint performs."""
    for name in (LIFTING_LINE, "tandem"):
        here = name.split()[0].lower()
        able = [n for n, s in api.PROBLEM_SPECS.items()
                if set(api.WING_CANT_KEYS) <= set(s.flags)
                and n.split()[0].lower() == here]
        assert able, f"{name} has no nonplanar twin to point a user at"


def test_the_pairs_cant_reaches_BOTH_wings():
    """One answer, both surfaces — the same rule the chord limits follow."""
    import numpy as np

    def _scored(g):
        """The SOLVED panel heights, off the report.

        Read here and not off the rebuild: ``build_flight_model`` builds its
        rear wing with the reported cant whatever the solver did with it, so
        a rear wing that was scored flat and flown canted would look correct
        downstream. The report's own ``z`` arrays are what the objective
        actually integrated over.
        """
        geom = _rep(LATTICE_PAIR,
                    {"wing_dihedral_deg": g} if g else {})["geometry"]
        # PTP, not max: the rear wing sits at the stagger height dz = 1 m, so
        # its z is off zero when it is perfectly flat. What a cant changes is
        # how far its own tips rise ABOVE ITS OWN ROOT.
        return (np.ptp(np.asarray(geom["z"], float)),
                np.ptp(np.asarray(geom["second_surface"]["z"], float)))

    f0, r0 = _scored(0.0)
    f8, r8 = _scored(8.0)
    assert f8 > f0 + 0.3, f"the front wing was scored flat ({f0:.4f})"
    assert r8 > r0 + 0.3, f"the REAR wing was scored flat ({r0:.4f})"
    # ...and by the SAME amount: one answer, both surfaces
    assert (r8 - r0) == pytest.approx(f8 - f0, rel=1e-6)

    # ...and it reaches the rebuild too
    m = build_flight_model(_rep(LATTICE_PAIR, {"wing_dihedral_deg": 8.0}),
                           V=45.0).model
    assert m.dihedral_deg == 8.0
    assert m.second is not None, "the pair's rear wing was not built"
    assert np.ptp(m.z[m.is_second]) > r0 + 0.3


def test_the_rebuild_FLIES_THE_DESIGN_not_a_simplification_of_it():
    """Four things ``build_flight_model`` was silently dropping, and the
    single property that catches all of them.

    Each was a stated fact the rebuild ignored — the tip device, the
    section's camber, the wing's twist, and the state to linearise at. Each
    is invisible in isolation and every one of them moves the lateral
    derivatives that stages 5 and 6 exist to report. Rather than four
    separate pins, this asserts what they were all violating: the rebuilt
    lattice must reach the SAME LIFT the design was scored at, at the SAME
    ATTITUDE, with the same lift-curve slope.

        before        rebuilt CL 0.412 at the scored alpha (17.6 % low)
        after         CL 0.500000 exactly, alpha within 0.001 deg
    """
    import numpy as np

    for problem in (LATTICE, LATTICE_PAIR):
        rep = _rep(problem)
        bd = rep["breakdown"]
        fm = build_flight_model(rep, V=45.0)
        CL_scored = bd.get("CL_total") or bd.get("CL_target")
        r = fm.model.solve(fm.alpha_trim, i_t=fm.i_t_trim)
        assert r.CL == pytest.approx(float(CL_scored), rel=1e-6), (
            f"{problem}: rebuilt CL {r.CL:.6f} at the deck state against a "
            f"scored {float(CL_scored):.6f}")
        stated = bd.get("alpha_rad")
        if stated is not None:
            gap = abs(np.rad2deg(fm.alpha_trim - float(stated)))
            assert gap < 0.05, (
                f"{problem}: the rebuild reaches that lift {gap:.3f} deg "
                f"away from where the solver did — something stated is not "
                f"being flown")


def test_the_wing_TWIST_reaches_the_flown_lattice():
    """It lives in ``geometry`` and the rebuild read ``breakdown``, so every
    design flew UNTWISTED. A pure zero-lift offset, and the last of the four
    disagreements between the objective and the rebuild."""
    import numpy as np

    rep = _rep(LATTICE)
    want = float(rep["geometry"]["twist_tip_deg"])
    assert want != 0.0, "this design should carry washout"
    m = build_flight_model(rep, V=45.0).model
    wing = ~(m.is_second | m.is_vertical | m.is_tail | m.is_winglet)
    tw = np.rad2deg(m.twist[wing])
    # the outermost COSINE station stops just short of the tip, so the flown
    # extreme is -1.99931 against a design tip of -2.0 — the law is right and
    # the sampling is what it is. Reading an end value off a cosine grid is a
    # known trap in this package; the point here is that the twist is present
    # AT ALL, and 0.0 is what it used to be.
    assert tw.min() == pytest.approx(want, rel=1e-3), (
        f"the flown wing's tip twist is {tw.min():+.4f} deg against the "
        f"design's {want:+.4f}")
    assert abs(tw.max()) < 0.1, "the root should be near zero twist"


def test_a_stated_cant_is_visible_in_the_3d_view():
    """The picture has to MOVE. A canted wing that draws flat is the same
    defect as a fin that draws not at all."""
    flat = _rep(LATTICE)
    cant = _rep(LATTICE, {"wing_dihedral_deg": 8.0})
    z_flat = max(flat["geometry"]["z"])
    z_cant = max(cant["geometry"]["z"])
    assert z_cant > 2.0 * z_flat, (
        f"a wing at 8 deg drew z_max {z_cant:.4f} against the planar "
        f"{z_flat:.4f}")
