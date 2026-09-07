"""Item 4 gate: closed-form wing CL_max and spanwise stall origin (stall.py).

The claim under test is that the critical-section stall point can be read off
TWO lifting-line solves instead of an alpha march, because the LLT is affine
in root incidence. So the gates are:

  - PARITY: an independent brentq march that solves
        max_y (Cl_y(alpha) - cl_max_sec(y)) = 0
    reproduces the closed-form alpha* (and the CL there) to 1e-12 on several
    planforms — including one with twist and one non-trapezoidal;
  - AFFINENESS: CL and Cl_y rebuilt from the two spanning solves match a
    THIRD, direct solve at an arbitrary alpha;
  - PHYSICS DIRECTION (no magic numbers): washout moves the stall origin
    INBOARD, strong taper moves it OUTBOARD, a locally thinner section
    (lower ceiling) drags the origin to itself, and sweeping the wing (which
    only reduces the section slope here) delays alpha*;
  - CEILING SOURCE: section_clmax interpolates the wide-alpha family the way
    PolarFamily.at does and REFUSES the censored thin end;
  - COST: the closed form is measured against a plain evaluate_geometry call
    and against the brentq march it replaces;
  - REGRESSION: FROZEN_LOD is untouched and objective.py does not import
    stall.py (checked in a fresh interpreter), so the BO objective's return
    value and cost are unchanged.
"""

import subprocess
import sys
import time

import numpy as np
import pytest
from scipy.optimize import brentq

from aerobo import geometry, stall
from aerobo.llt import cosine_stations, elliptic_chord, solve_llt
from aerobo.objective import Problem, evaluate_geometry, objective
from aerobo.polar import PolarFamily, TablePolar, default_polar, stall_points
from aerobo.stall import (
    default_stall_family,
    evaluate_stall,
    section_clmax,
    wing_clmax,
)

# same frozen literal as tests/test_physics_flags.py / test_stall_polar.py
FROZEN_LOD = 34.761763013754106
FROZEN_PROBE = np.array([0.4, 1.0, -2.0])

B = 10.0
S = 10.0
N = 60
POL = default_polar()          # NACA 2412 Re 1e6 (XFOIL) — the Tier A section
A_SEC = POL.a_lin
AL0 = POL.alpha_L0
CLMAX_2412 = 1.5307            # tabulated 2-D peak, t/c = 0.12 (Item 3)

try:
    STALL_FAMILY = default_stall_family()
except FileNotFoundError:  # pragma: no cover - data files are committed
    STALL_FAMILY = None

needs_stall = pytest.mark.skipif(
    STALL_FAMILY is None, reason="naca24XX_re1e6_stall.pol family missing")


def _trapezoid(taper: float, tw_root: float = 0.0, tw_tip: float = 0.0):
    """(c, twist_rad) at the N cosine stations — the codebase's own sampling."""
    wing = geometry.Wing(b=B, S=S, taper=taper,
                         twist_root_deg=tw_root, twist_tip_deg=tw_tip)
    _, c, twist = wing.sample(N)
    return c, twist


def _elliptic():
    _, y = cosine_stations(N, B)
    return elliptic_chord(y, B, S), np.zeros(N)


# 4 planforms: rectangular, strongly tapered, tapered + washout, elliptic
PLANFORMS = {
    "rect": _trapezoid(1.0),
    "taper02": _trapezoid(0.2),
    "taper05_washout": _trapezoid(0.5, 1.0, -4.0),
    "elliptic": _elliptic(),
}


def _march_alpha(c, twist, cl_max, a=A_SEC, alpha_L0=AL0) -> float:
    """Independent oracle: brentq on max_y(Cl_y(alpha) - cl_max_sec) = 0.

    The envelope of the (affine) strip lines is convex and increasing here,
    so the root is unique; nothing about the closed form is reused."""
    def resid(alpha: float) -> float:
        res = solve_llt(B, c, alpha + twist, a, alpha_L0)
        return float(np.max(res.Cl_y - cl_max))

    return float(brentq(resid, -0.5, 1.0, xtol=1e-15, rtol=8.9e-16))


# ---------------- 1. parity with an independent alpha march ----------------


@pytest.mark.parametrize("name", sorted(PLANFORMS))
def test_closed_form_matches_the_brentq_alpha_march(name):
    c, twist = PLANFORMS[name]
    ws = wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    alpha_march = _march_alpha(c, twist, CLMAX_2412)

    assert np.deg2rad(ws.alpha_star_deg) == pytest.approx(alpha_march, abs=1e-12)
    # and the lift AT that incidence, from a direct solve the closed form
    # never sees
    direct = solve_llt(B, c, alpha_march + twist, A_SEC, AL0)
    assert direct.CL == pytest.approx(ws.CL_max, abs=1e-12)
    # the marched loading touches the ceiling exactly at the reported origin
    # (untwisted planforms are mirror-symmetric, so the peak is a two-station
    # tie broken by roundoff — compare |eta|, not the raw index)
    assert direct.Cl_y[ws.i_stall] == pytest.approx(CLMAX_2412, abs=1e-12)
    assert direct.Cl_y.max() - CLMAX_2412 < 1e-12
    # The stall ORIGIN is only well posed when the margin is not near-uniform.
    # The elliptic planform against a constant ceiling is the degenerate case
    # (every strip reaches the ceiling within roundoff of alpha*), so the
    # closed form and the march legitimately break the tie on different
    # strips; the flag must catch that, and only the non-degenerate planforms
    # are held to origin parity.
    if ws.stall_degenerate:
        assert name == "elliptic"
        assert ws.n_near_stall > 0.10 * N
    else:
        i_march = int(np.argmax(direct.Cl_y))
        assert abs(2.0 * ws.y[i_march] / B) == pytest.approx(ws.eta_stall,
                                                             abs=1e-12)


@needs_stall
def test_parity_with_a_spanwise_varying_ceiling():
    """A non-uniform ceiling (thicker root, thinner tip) is the general case:
    the march must still agree with the closed form."""
    c, twist = PLANFORMS["taper02"]
    _, y = cosine_stations(N, B)
    tc_y = 0.18 - 0.06 * np.abs(2.0 * y / B)          # 0.18 root -> 0.12 tip
    cl_max_y = section_clmax(tc_y, family=STALL_FAMILY)
    assert cl_max_y.shape == (N,)

    ws = wing_clmax(B, c, twist, cl_max_y, a=A_SEC, alpha_L0=AL0)
    alpha_march = _march_alpha(c, twist, cl_max_y)
    assert np.deg2rad(ws.alpha_star_deg) == pytest.approx(alpha_march, abs=1e-12)


# ---------------- 2. affineness sanity ----------------


@pytest.mark.parametrize("name", sorted(PLANFORMS))
def test_two_solves_reproduce_a_third_direct_solve(name):
    """CL(alpha) = CL_0 + alpha*CL_alpha and Cl_y(alpha) = p + alpha*q are the
    whole trick; a third solve at an arbitrary alpha must land on them."""
    c, twist = PLANFORMS[name]
    ws = wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)

    for alpha in (-0.037, 0.0, 0.1234, 0.4):
        direct = solve_llt(B, c, alpha + twist, A_SEC, AL0)
        assert direct.CL == pytest.approx(ws.CL_0 + alpha * ws.CL_alpha, abs=1e-13)
        assert np.max(np.abs(direct.Cl_y - (ws.p + alpha * ws.q))) < 1e-13


def test_margin_is_zero_at_the_origin_and_nonnegative_elsewhere():
    c, twist = PLANFORMS["taper02"]
    ws = wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    assert ws.margin_y[ws.i_stall] == pytest.approx(0.0, abs=1e-14)
    assert ws.margin_y.min() >= -1e-14
    assert np.argmin(ws.margin_y) == ws.i_stall
    # the per-strip stall incidences are all >= alpha*, equal at the origin
    assert ws.alpha_stall_y_deg.min() == pytest.approx(ws.alpha_star_deg)
    assert ws.eta_stall == pytest.approx(abs(ws.eta[ws.i_stall]))


# ---------------- 3. physics directions (no magic numbers) ----------------


def test_washout_moves_the_stall_origin_inboard():
    """Twisting the tip nose-down unloads it: the first section to reach its
    ceiling must move TOWARD the root, at the same planform."""
    base = wing_clmax(B, *_trapezoid(0.2), cl_max_sec=CLMAX_2412,
                      a=A_SEC, alpha_L0=AL0)
    for tip in (-2.0, -4.0, -6.0):
        washed = wing_clmax(B, *_trapezoid(0.2, 0.0, tip),
                            cl_max_sec=CLMAX_2412, a=A_SEC, alpha_L0=AL0)
        assert washed.eta_stall < base.eta_stall
    # monotone in washout magnitude, and it buys lift (the outboard strips
    # are no longer the limiter)
    e2 = wing_clmax(B, *_trapezoid(0.2, 0.0, -2.0), cl_max_sec=CLMAX_2412,
                    a=A_SEC, alpha_L0=AL0)
    e6 = wing_clmax(B, *_trapezoid(0.2, 0.0, -6.0), cl_max_sec=CLMAX_2412,
                    a=A_SEC, alpha_L0=AL0)
    assert e6.eta_stall < e2.eta_stall
    assert base.CL_max < e2.CL_max


def test_strong_taper_moves_the_stall_origin_outboard():
    """Classic result: the tip chord shrinks faster than the local lift, so a
    strongly tapered wing (small taper ratio) stalls outboard while the
    rectangular one stalls at the root."""
    etas = [wing_clmax(B, *_trapezoid(lam), cl_max_sec=CLMAX_2412,
                       a=A_SEC, alpha_L0=AL0).eta_stall
            for lam in (0.2, 0.35, 0.5, 1.0)]
    assert etas[0] > etas[-1]                      # taper 0.2 vs rectangular
    assert all(hi > lo for hi, lo in zip(etas[:-1], etas[1:]))   # monotone


def test_a_locally_lower_ceiling_captures_the_origin():
    """The ceiling, not just the loading, decides: drop cl_max on one strip
    of the rectangular wing (which otherwise stalls at the root) and the
    origin must jump to that strip."""
    c, twist = PLANFORMS["rect"]
    base = wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    ceiling = np.full(N, CLMAX_2412)
    victim = 15                                    # an outboard strip
    assert base.i_stall != victim
    # put its ceiling BELOW what it already carries when the wing stalls, so
    # it is guaranteed to be reached first
    ceiling[victim] = 0.9 * base.Cl_y_stall[victim]
    spoilt = wing_clmax(B, c, twist, ceiling, a=A_SEC, alpha_L0=AL0)
    assert spoilt.i_stall == victim
    assert spoilt.eta_stall > base.eta_stall
    assert spoilt.CL_max < base.CL_max


def test_scalar_and_broadcast_ceilings_agree():
    """A uniform section given as a scalar and as an (N,) array of the same
    value must produce the identical answer."""
    c, twist = PLANFORMS["taper05_washout"]
    a = wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    b = wing_clmax(B, c, twist, np.full(N, CLMAX_2412), a=A_SEC, alpha_L0=AL0)
    assert a.CL_max == b.CL_max
    assert a.i_stall == b.i_stall
    assert a.alpha_star_deg == b.alpha_star_deg


def test_sweep_delays_the_stall_incidence():
    """Simple-sweep theory only cuts the section slope here (the LLT stays
    unswept), so a swept wing needs MORE root incidence to reach the same
    ceiling."""
    prob = Problem()
    c, twist = PLANFORMS["taper05_washout"]
    straight = evaluate_stall(c, twist, prob, cl_max_sec=CLMAX_2412)
    swept = evaluate_stall(c, twist, prob, cl_max_sec=CLMAX_2412, sweep_deg=30.0)
    assert swept.alpha_star_deg > straight.alpha_star_deg


# ---------------- 4. the 2-D ceiling source ----------------


@needs_stall
def test_section_clmax_is_exact_at_members_and_linear_between():
    pts = stall_points(STALL_FAMILY)
    for tc in (0.09, 0.12, 0.15, 0.18):
        assert section_clmax(tc, family=STALL_FAMILY) == pts[tc].cl_max
    mid = 0.5 * (pts[0.12].cl_max + pts[0.15].cl_max)
    assert section_clmax(0.135, family=STALL_FAMILY) == pytest.approx(mid)
    assert section_clmax(0.12, family=STALL_FAMILY) == pytest.approx(1.5307)


@needs_stall
def test_section_clmax_refuses_the_censored_thin_end():
    """t/c = 0.06 is right-censored at Re 1e6 (its BL diverges before stall),
    so it must not anchor the interpolation — and a t/c below the resolved
    floor has no cl_max at all, even though geometry.TC_BOUNDS allows 0.08."""
    assert stall_points(STALL_FAMILY)[0.06].censored
    assert geometry.TC_BOUNDS[0] < 0.09
    with pytest.raises(ValueError, match="RESOLVED"):
        section_clmax(0.08, family=STALL_FAMILY)
    with pytest.raises(ValueError, match="RESOLVED"):
        section_clmax(0.19, family=STALL_FAMILY)
    # the censored member's (lower-bound) peak never leaks into the blend
    assert section_clmax(0.09, family=STALL_FAMILY) > 1.4


def test_section_clmax_needs_at_least_one_resolved_member():
    a = np.arange(-6.0, 9.0 + 1e-9, 0.5)
    cl = 1.5 - 0.006 * (a - 15.0) ** 2              # still climbing: censored
    z = np.zeros_like(a)
    fam = PolarFamily({
        0.06: TablePolar(alpha_deg=a, CL=cl, CD=z + 0.01, CM=z, name="s6"),
        0.09: TablePolar(alpha_deg=a, CL=cl + 0.05, CD=z + 0.01, CM=z, name="s9"),
    })
    with pytest.raises(ValueError, match="censored"):
        section_clmax(0.08, family=fam)


# ---------------- 5. problem-level entry point ----------------


@needs_stall
def test_evaluate_stall_matches_the_kernel_on_the_tier_a_problem():
    prob = Problem()
    c, twist = PLANFORMS["taper02"]
    via_prob = evaluate_stall(c, twist, prob, tc=0.12, stall_family=STALL_FAMILY)
    direct = wing_clmax(prob.b, c, twist, CLMAX_2412, a=POL.a_lin,
                        alpha_L0=POL.alpha_L0)
    assert via_prob.CL_max == direct.CL_max
    assert via_prob.eta_stall == direct.eta_stall


def test_evaluate_stall_wants_exactly_one_ceiling_source():
    prob = Problem()
    c, twist = PLANFORMS["rect"]
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_stall(c, twist, prob)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_stall(c, twist, prob, cl_max_sec=1.5, tc=0.12)


def test_evaluate_stall_refuses_phase5_skins():
    """The kernel solves the PLAIN free-air lifting line, so it must refuse
    a Problem that carries a ground or slipstream skin rather than silently
    returning free-air stall numbers next to a skin-affected L/D."""
    c, twist = PLANFORMS["rect"]
    with pytest.raises(NotImplementedError, match="ground_h_m"):
        evaluate_stall(c, twist, Problem(ground_h_m=1.0), cl_max_sec=1.5)
    with pytest.raises(NotImplementedError, match="slipstream"):
        evaluate_stall(c, twist, Problem(slipstream=object()), cl_max_sec=1.5)


def test_evaluate_stall_checks_grid_matches_problem_N():
    """b comes from the Problem, the grid from c, so a chord sampled at a
    different N than the Problem declares would silently solve another wing."""
    c, twist = PLANFORMS["rect"]                       # sampled at N
    with pytest.raises(ValueError, match="stations"):
        evaluate_stall(c[:-1], twist[:-1], Problem(N=N), cl_max_sec=1.5)


def test_stall_origin_is_flagged_degenerate_on_the_elliptic_wing():
    """Near-uniform margin => the origin is a roundoff tie, not a location.
    The elliptic planform against a constant ceiling is the canonical case;
    a strongly tapered wing with the same ceiling is not degenerate."""
    c_ell, tw_ell = PLANFORMS["elliptic"]
    ell = wing_clmax(B, c_ell, tw_ell, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    assert ell.stall_degenerate
    assert ell.n_near_stall > 0.10 * N

    c_t, tw_t = PLANFORMS["taper02"]
    tap = wing_clmax(B, c_t, tw_t, CLMAX_2412, a=A_SEC, alpha_L0=AL0)
    assert not tap.stall_degenerate
    # the tapered wing has a genuine, localised stall origin: far fewer strips
    # crowd alpha* than on the elliptic wing (whole span within roundoff)
    assert tap.n_near_stall < 0.5 * ell.n_near_stall


@needs_stall
def test_evaluate_stall_serves_the_free_planform_wing():
    """aircraft.py's pattern: b and S are design variables and the trim
    Problem is built locally — the same call must work, and the answer must
    track the geometry (larger AR at the same S loads the tip harder relative
    to its chord, so the origin does not silently stay put)."""
    out = {}
    for b, S_ in ((25.0, 18.0), (12.0, 18.0)):
        wing = geometry.Wing(b=b, S=S_, taper=0.45,
                             twist_root_deg=1.0, twist_tip_deg=-2.0, tc=0.15)
        _, c, twist = wing.sample(N)
        aero_prob = Problem(b=b, S=S_, N=N, polar=POL)
        ws = evaluate_stall(c, twist, aero_prob, tc=wing.tc,
                            stall_family=STALL_FAMILY)
        assert ws.S == pytest.approx(S_, rel=2e-3)   # trapezoid on cosine grid
        assert ws.AR == pytest.approx(b**2 / ws.S)
        assert 0.0 < ws.eta_stall <= 1.0
        assert 1.0 < ws.CL_max < 1.6
        out[b] = ws
    # high AR -> smaller induced angle -> the outboard relief shrinks and the
    # wing reaches its ceiling at a LOWER root incidence
    assert out[25.0].alpha_star_deg < out[12.0].alpha_star_deg


# ---------------- 6. error contract ----------------


def test_wing_clmax_rejects_a_nonpositive_or_nonfinite_ceiling():
    c, twist = PLANFORMS["rect"]
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            wing_clmax(B, c, twist, bad, a=A_SEC, alpha_L0=AL0)


def test_wing_clmax_rejects_a_wing_that_never_reaches_its_ceiling(monkeypatch):
    """Guard branch, forced: if the loading does not respond to root incidence
    (q = 0 at every strip) there is no stall point to report. No real planform
    does that with a positive section slope, so the degenerate LLT is injected
    by returning the SAME solve for both incidences."""
    c, twist = PLANFORMS["rect"]
    frozen = solve_llt(B, c, twist, A_SEC, AL0)
    monkeypatch.setattr(stall, "solve_llt", lambda *a, **k: frozen)
    with pytest.raises(ValueError, match="no strip gains lift"):
        wing_clmax(B, c, twist, CLMAX_2412, a=A_SEC, alpha_L0=AL0)


# ---------------- 7. cost ----------------


def _time_ms(fn, n: int = 300) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


@needs_stall
def test_cost_is_below_one_objective_evaluation(capsys):
    """Measured overhead of the diagnostic, against the objective's own
    physics core and against the brentq march it replaces."""
    prob = Problem()
    c, twist = _trapezoid(0.4, 1.0, -2.0)

    t_eval = _time_ms(lambda: evaluate_geometry(c, twist, prob))
    t_stall = _time_ms(lambda: evaluate_stall(c, twist, prob, tc=0.12,
                                              stall_family=STALL_FAMILY))
    t_const = _time_ms(lambda: evaluate_stall(c, twist, prob,
                                              cl_max_sec=CLMAX_2412))
    t_march = _time_ms(lambda: _march_alpha(c, twist, CLMAX_2412), n=100)

    with capsys.disabled():
        print(f"\n[cost] evaluate_geometry      {t_eval:.4f} ms")
        print(f"[cost] evaluate_stall(tc)     {t_stall:.4f} ms "
              f"({t_stall / t_eval:.2f} x)")
        print(f"[cost] evaluate_stall(const)  {t_const:.4f} ms "
              f"({t_const / t_eval:.2f} x)")
        print(f"[cost] brentq alpha march     {t_march:.4f} ms "
              f"({t_march / t_eval:.2f} x)")

    # two solves and a division: cheaper than the trimmed objective call it
    # sits beside, and several times cheaper than the march it replaces
    assert t_stall < 1.5 * t_eval
    assert t_stall < 0.5 * t_march


# ---------------- 8. the objective is untouched ----------------


def test_frozen_lod_unchanged():
    assert objective(FROZEN_PROBE, Problem()) == FROZEN_LOD      # ==, no tol


def test_objective_does_not_import_stall():
    """Checked in a FRESH interpreter: importing the objective must not drag
    in the stall diagnostic, so neither the BO objective's return value nor
    its per-evaluation cost can depend on this module."""
    code = ("import sys; import aerobo.objective; "
            "sys.exit(1 if 'aerobo.stall' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
