"""One aeroplane, two instruments, opposite answers about its spiral.

With the fin switched off, ``Cn_beta`` is exactly -0.0 and:

* ``api.dihedral_for_spiral`` REFUSED the design — "this aeroplane is
  directionally unstable and a positive spiral margin on it is arithmetic,
  not stability";
* ``gui.v4.modes.classify`` named a mode "spiral" at -0.02221 and the
  Flight stage printed it CONVERGES, HALVING IN 31.2 s.

The second is not a rounding of the first, it is a different claim: with no
yaw stiffness the yaw equation decouples, so the slow lateral real root is
a sideslip subsidence, not a spiral. Naming it invents the mode and then
reports it stable.

And it fails the other way too, which is why the fix cannot simply be to
drop the mode. `hydrofoil + elevator` at its box centre has ``Cn_beta``
-0.19599 and a spiral CRITERION of +0.02016 — convergent on paper — while
the eigenvalue is +0.44979 and DOUBLES IN 1.54 s. That divergence is real
and must stay on the panel.

So: one shared predicate (``dynamics.weathercocks``), asked by the criterion
and by the namer; the root is always reported, and only its NAME changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, dynamics as dyn, flightmodel, sixdof as sd  # noqa: E402
from gui.v4 import modes as md                                     # noqa: E402

NAME = "tail [free height, designed tail] + free chord law"


def _flown(flags: dict, problem: str = NAME):
    """(report, deck, named modes) for a design, through the shipped bridge."""
    spec = api.PROBLEM_SPECS[problem]
    built = spec.build({}, flags, None)
    bnds = built.problem.bounds
    x = 0.5 * (bnds[:, 0] + bnds[:, 1])
    rep = api.design_report(api.RunConfig(problem_name=problem, flags=flags), x)
    fm = flightmodel.build_flight_model(rep)
    V = float((rep["breakdown"] or {}).get("V") or flightmodel.DEFAULT_V_MS)
    st, ac = sd.trim_level(fm.aircraft, V=V)
    named = md.classify(sd.linearise(ac, st), float(st.V),
                        Cn_beta=ac.deck.Cn_beta)
    return rep, fm.deck, named


_FIN = {"tail_type": "conventional", "fin": True, "fuselage_diameter_m": 0.22}
_NO_FIN = {"tail_type": "conventional", "fin": False,
           "fuselage_diameter_m": 0.22}


# ------------------------------------------------------- the shared predicate

def test_one_definition_of_weathercocking():
    assert dyn.weathercocks(0.11) is True
    assert dyn.weathercocks(0.0) is False        # the finless case exactly
    assert dyn.weathercocks(-0.0) is False
    assert dyn.weathercocks(-0.196) is False
    assert dyn.weathercocks(None) is False
    assert dyn.weathercocks(float("nan")) is False


def test_the_criterion_asks_that_same_question():
    """``wing_score.spiral_refusal`` must not keep a second copy of it."""
    from aerobo import wing_score

    for cnb in (0.11, 0.0, -0.0, -0.196):
        raw = {"Cn_beta": cnb, "spiral_margin": +0.02}
        refused = bool(wing_score.spiral_refusal(raw))
        assert refused is (not dyn.weathercocks(cnb)), cnb


# ------------------------------------------------------------- the two agree

def test_a_finless_design_is_no_longer_told_its_spiral_converges():
    rep, deck, named = _flown(_NO_FIN)
    assert api.dihedral_for_spiral(rep)["status"] == "refused"
    assert deck.Cn_beta <= 0.0
    assert "spiral" not in named, sorted(named)
    # ...and the root is not hidden: it is there under its honest name
    m, name = md.find(named, "spiral")
    assert m is not None and name == md.SLOW_LATERAL
    assert md.why_not_a_spiral(deck.Cn_beta)


def test_the_finned_twin_is_unchanged_and_still_has_a_spiral():
    """The control. Without it the test above passes on a namer that
    dropped the mode for everybody."""
    rep, deck, named = _flown(_FIN)
    assert api.dihedral_for_spiral(rep)["status"] != "refused"
    assert deck.Cn_beta > 0.0
    assert "spiral" in named and "dutch roll" in named
    assert md.find(named, "spiral")[1] == "spiral"
    assert md.why_not_a_spiral(deck.Cn_beta) == ""


def test_a_real_divergence_survives_the_rename():
    """`hydrofoil + elevator`: refused BY THE CRITERION and genuinely
    divergent by the eigenvalue. The panel must still carry the number."""
    rep, deck, named = _flown({}, "hydrofoil + elevator")
    assert deck.Cn_beta < 0.0
    assert dyn.spiral_margin_of(deck) > 0.0, "the criterion reads convergent"
    m, name = md.find(named, "spiral")
    assert name == md.SLOW_LATERAL
    assert m is not None and m.diverges, m
    assert m.double_or_half_s < 3.0, m.double_or_half_s


def test_omitting_the_yaw_stiffness_names_exactly_what_it_always_did():
    """No existing caller moves: the argument is optional and absent means
    'do not know', which is the previous behaviour verbatim."""
    _rep, _deck, _named = _flown(_NO_FIN)
    spec = api.PROBLEM_SPECS[NAME]
    built = spec.build({}, _NO_FIN, None)
    bnds = built.problem.bounds
    x = 0.5 * (bnds[:, 0] + bnds[:, 1])
    rep = api.design_report(api.RunConfig(problem_name=NAME, flags=_NO_FIN), x)
    fm = flightmodel.build_flight_model(rep)
    V = float(rep["breakdown"]["V"])
    st, ac = sd.trim_level(fm.aircraft, V=V)
    J = sd.linearise(ac, st)
    old = md.classify(J, float(st.V))
    new = md.classify(J, float(st.V), Cn_beta=ac.deck.Cn_beta)
    assert "spiral" in old and "spiral" not in new
    assert old["spiral"].real == new[md.SLOW_LATERAL].real


# --------------------------------------------------- and stage 6 says it

def _modes_panel(flags: dict) -> list[str]:
    """What stage 6's Modes panel actually draws for a design."""
    from gui.v4 import app as v4app

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", flags=flags, budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    ctx.render("controls", "derivatives")
    ctx.select("flight", "fly")
    ctx.act("flight_run", True)
    for _ in range(5):
        ctx.act("flight_advance", 1 / 30)
    ctx.render("flight", "modes")
    return [t for e in ctx.views[("flight", "modes")].descendants()
            if (t := (getattr(e, "text", "") or ""))]


def test_the_flight_panel_stops_reporting_a_finless_spiral():
    lines = _modes_panel({"fin": False})
    text = " ".join(lines)
    assert md.SLOW_LATERAL in lines, lines
    assert "does NOT weathercock" in text
    # the row that used to say "spiral ... half in 31.2 s" is gone
    assert "spiral" not in lines, lines
    # ...and the ROOT is still on the panel, with its time
    assert any("31." in t for t in lines), lines


def test_the_finned_design_s_panel_is_unchanged():
    lines = _modes_panel({"fin": True})
    assert "spiral" in lines and "dutch roll" in lines, lines
    assert md.SLOW_LATERAL not in lines
    assert "does NOT weathercock" not in " ".join(lines)


def _manoeuvre_rows(flags: dict) -> str:
    """The Fly tab's "Stability manoeuvres" block, as text."""
    from gui.v4 import app as v4app

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", flags=flags, budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    ctx.render("controls", "derivatives")
    ctx.select("flight", "fly")
    ctx.act("flight_run", True)
    ctx.act("flight_advance", 1 / 30)
    ctx.render("flight", "fly")
    # some elements carry a non-str ``text`` (a class, on a factory), so
    # coerce rather than join blindly
    return " ".join(
        str(t) for e in ctx.views[("flight", "fly")].descendants()
        if isinstance(t := (getattr(e, "text", "") or ""), str) and t)


def test_the_spiral_manoeuvre_says_which_root_it_is_quoting():
    """The button stays flyable — it is 10 deg of bank and nothing else —
    but the mode beside it is not a spiral on a design with no yaw
    stiffness, and the row says so instead of printing a spiral's numbers
    under a spiral's name."""
    assert f"[{md.SLOW_LATERAL}]" in _manoeuvre_rows({"fin": False})


def test_and_says_nothing_extra_where_it_IS_the_spiral():
    assert f"[{md.SLOW_LATERAL}]" not in _manoeuvre_rows({"fin": True})
