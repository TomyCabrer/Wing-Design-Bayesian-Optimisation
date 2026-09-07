"""Item 4 gates: Tier B tandem two-surface coupling (src/aerobo/tandem.py).

1. Kernel normalisation [ANALYTIC]: the panel-horseshoe influence kernel,
   evaluated at the source wing's own lifting line, reproduces the Fourier
   LLT self-induced angle (elliptic loading -> constant CL/(pi AR)); far
   downstream the downwash doubles (Trefftz limit).
2. Decoupling limit: huge vertical gap -> each wing reproduces the isolated
   single-wing LLT solution to 1e-8.
3. Munk's stagger theorem [ANALYTIC, Munk 1921]: at FIXED circulation
   distributions the total mutual induced drag is independent of streamwise
   stagger dx — exact for the panel-horseshoe system, verified to 1e-12.
4. Biplane gap trend [ANALYTIC, Prandtl 1924]: equal elliptic wings at zero
   stagger — interference factor sigma matches Prandtl's multiplane formula
   sigma = (1 - 0.66 h/b)/(1.05 + 3.7 h/b) to 2 % inside its validity range,
   is monotone decreasing in gap, -> 1 (monoplane of double lift) as h -> 0
   and -> 0 (sum of isolated wings) as h -> inf.
5. Coupled-solver physics: front wing gains lift (rear-wing upwash, induced
   THRUST CDi_mut_front < 0), rear wing loses lift (front-wing downwash,
   CDi_mut_rear > 0); translation invariance front<->rear.
6. Tier B objective: 10-D vector, trim to system CL_target, -100.0 contract.
"""

import numpy as np
import pytest

from aerobo.llt import cosine_stations, elliptic_chord, solve_llt
from aerobo.objective import PENALTY
from aerobo.tandem import (
    Surface,
    TandemProblem,
    VortexSystem,
    evaluate_tandem,
    mutual_cdi,
    objective_tandem,
    solve_tandem,
    w_influence,
)

B = 10.0
AR = 10.0
S = B**2 / AR
N = 60


def _elliptic_gamma(alpha_deg: float = 5.0):
    _, y = cosine_stations(N, B)
    c = elliptic_chord(y, B, S)
    res = solve_llt(B, c, np.full(N, np.deg2rad(alpha_deg)))
    return y, c, res


# ------------------------------------------------------------ 1. kernel gates

def test_kernel_matches_llt_self_downwash_elliptic():
    """[ANALYTIC] elliptic loading -> eps = -CL/(pi AR) at the lifting line."""
    y, c, res = _elliptic_gamma()
    sys0 = VortexSystem(b=B, N=N)
    pts = np.column_stack([np.zeros(N), y, np.full(N, 1e-3 * B)])
    eps = w_influence(pts, sys0) @ res.Gamma          # V = 1
    mid = np.abs(2 * y / B) < 0.8                     # away from tip panels
    expected = -res.CL / (np.pi * res.AR)
    assert np.all(eps[mid] < 0.0)                     # downwash
    assert np.max(np.abs(eps[mid] - expected)) < 0.01 * abs(expected)


def test_kernel_far_wake_downwash_doubles():
    """[ANALYTIC] Trefftz limit: far-downstream downwash = 2x at-the-line."""
    y, c, res = _elliptic_gamma()
    sys0 = VortexSystem(b=B, N=N)
    mid = np.abs(2 * y / B) < 0.8
    near = w_influence(
        np.column_stack([np.zeros(N), y, np.full(N, 1e-3 * B)]), sys0
    ) @ res.Gamma
    far = w_influence(
        np.column_stack([np.full(N, 50 * B), y, np.zeros(N)]), sys0
    ) @ res.Gamma
    assert far[mid].mean() / near[mid].mean() == pytest.approx(2.0, abs=0.01)


def test_kernel_far_upstream_vanishes():
    y, c, res = _elliptic_gamma()
    sys0 = VortexSystem(b=B, N=N)
    up = w_influence(
        np.column_stack([np.full(N, -50 * B), y, np.zeros(N)]), sys0
    ) @ res.Gamma
    near = -res.CL / (np.pi * res.AR)
    assert np.max(np.abs(up)) < 1e-3 * abs(near)


# ------------------------------------------------------- 2. decoupling limit

def test_decoupling_limit_reproduces_single_wing():
    """Huge vertical gap -> coupled solve == two isolated LLT solves."""
    _, y = cosine_stations(N, B)
    c_ell = elliptic_chord(y, B, S)
    c_rect = np.full(N, S / B)
    al_f = np.full(N, np.deg2rad(5.0))
    al_r = np.deg2rad(3.0) - np.deg2rad(2.0) * np.abs(2 * y / B)  # twisted

    tr = solve_tandem(
        Surface(b=B, c=c_ell, alpha_geo=al_f),
        Surface(b=B, c=c_rect, alpha_geo=al_r, x=5.0, z=1e4 * B),
    )
    iso_f = solve_llt(B, c_ell, al_f)
    iso_r = solve_llt(B, c_rect, al_r)

    assert tr.front.CL == pytest.approx(iso_f.CL, rel=1e-8)
    assert tr.rear.CL == pytest.approx(iso_r.CL, rel=1e-8)
    assert tr.front.CDi == pytest.approx(iso_f.CDi, rel=1e-8)
    assert tr.rear.CDi == pytest.approx(iso_r.CDi, rel=1e-8)
    assert tr.front.e == pytest.approx(iso_f.e, rel=1e-8)
    assert np.max(np.abs(tr.eps_rear)) < 1e-10
    assert tr.CDi_mut == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------- 3. Munk stagger theorem

def test_munk_stagger_invariance():
    """[ANALYTIC] Munk (1921): total mutual induced drag of two FIXED
    circulation distributions is independent of streamwise stagger."""
    y, c, res = _elliptic_gamma()
    Gam1 = res.Gamma
    Gam2 = res.Gamma * (1.0 + 0.3 * (2 * y / B) ** 2)   # different loading
    sums, rears = [], []
    for dx in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        sA = VortexSystem(b=B, N=N, x=0.0, z=0.0)
        sB = VortexSystem(b=B, N=N, x=dx, z=2.0)
        cdi_a, cdi_b = mutual_cdi(sA, sB, Gam1, Gam2, V=1.0, Sref=2 * S)
        sums.append(cdi_a + cdi_b)
        rears.append(cdi_b)
    sums = np.array(sums)
    spread = (sums.max() - sums.min()) / abs(sums.mean())
    assert spread < 1e-12                               # measured ~3e-16
    # the test has teeth: the individual terms DO move with stagger
    assert (max(rears) - min(rears)) > 0.1 * abs(sums.mean())


# ------------------------------------------------------- 4. biplane gap gate

def test_biplane_interference_vs_prandtl():
    """[ANALYTIC] equal elliptic wings, zero stagger: sigma(h/b) matches
    Prandtl's multiplane interference formula to 2 % and has the correct
    monoplane / isolated limits; monotone decreasing in gap."""
    y, c, res = _elliptic_gamma()
    self_cdi = res.CDi * res.S / (2 * S)                # one wing, on Sref
    gaps = np.array([0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 5.0]) * B
    sigma = []
    for h in gaps:
        cdi_a, cdi_b = mutual_cdi(
            VortexSystem(b=B, N=N), VortexSystem(b=B, N=N, z=h),
            res.Gamma, res.Gamma, V=1.0, Sref=2 * S,
        )
        sigma.append((cdi_a + cdi_b) / (2.0 * self_cdi))
    sigma = np.array(sigma)

    assert np.all(np.diff(sigma) < 0)                   # monotone decreasing
    assert sigma[0] > 0.85                              # -> monoplane 2x-lift
    assert sigma[-1] < 0.01                             # -> isolated wings
    # Prandtl (1924): sigma = (1 - 0.66 h/b)/(1.05 + 3.7 h/b), 0.05<=h/b<=0.5
    for h, s_num in zip(gaps, sigma):
        hb = h / B
        if 0.05 <= hb <= 0.5:
            s_prandtl = (1 - 0.66 * hb) / (1.05 + 3.7 * hb)
            assert s_num == pytest.approx(s_prandtl, rel=0.02)


# ------------------------------------------------- 5. coupled-solver physics

def _tandem_pair(dx=5.0, dz=1.0):
    _, y = cosine_stations(N, B)
    c = np.full(N, S / B)
    al = np.full(N, np.deg2rad(5.0))
    front = Surface(b=B, c=c, alpha_geo=al)
    rear = Surface(b=B, c=c, alpha_geo=al, x=dx, z=dz)
    return front, rear, solve_llt(B, c, al)


def test_tandem_front_gains_rear_loses():
    """Core Tier B effect: front-wing downwash penalises the rear wing;
    the rear wing's bound vortex gives the front wing upwash (induced
    thrust — reference lift_drag_strip.m 'book Fig 2.16')."""
    front, rear, iso = _tandem_pair()
    tr = solve_tandem(front, rear)
    assert tr.rear.CL < iso.CL                # downwash unloads the rear
    assert tr.front.CL > iso.CL               # upwash loads the front
    assert np.mean(tr.eps_rear) < 0.0         # eps21 < 0 (downwash), cf. MATLAB
    assert np.mean(tr.eps_front) > 0.0        # eps12 upwash on front
    assert tr.CDi_mut_rear > 0.0
    assert tr.CDi_mut_front < 0.0             # induced thrust on the front wing
    assert tr.CDi_mut > 0.0                   # net system penalty


def test_tandem_translation_invariance():
    """(W1 at origin, W2 at (dx,dz)) == (W2 at origin, W1 at (-dx,-dz))."""
    _, y = cosine_stations(N, B)
    c1 = elliptic_chord(y, B, S)
    c2 = np.full(N, S / B)
    al1 = np.full(N, np.deg2rad(5.0))
    al2 = np.full(N, np.deg2rad(3.0))
    t1 = solve_tandem(Surface(b=B, c=c1, alpha_geo=al1),
                      Surface(b=B, c=c2, alpha_geo=al2, x=5.0, z=1.0))
    t2 = solve_tandem(Surface(b=B, c=c2, alpha_geo=al2, x=0.0, z=0.0),
                      Surface(b=B, c=c1, alpha_geo=al1, x=-5.0, z=-1.0))
    assert t1.front.CL == pytest.approx(t2.rear.CL, rel=1e-12)
    assert t1.rear.CL == pytest.approx(t2.front.CL, rel=1e-12)
    assert t1.CDi_total == pytest.approx(t2.CDi_total, rel=1e-12)
    assert t1.CDi_mut_front == pytest.approx(t2.CDi_mut_rear, rel=1e-12)


def test_pure_streamwise_separation_never_decouples():
    """Physics note enforced: at dz = 0 the rear wing sits in the front
    wing's trailing wake at ANY dx — downwash tends to the doubled Trefftz
    value, not to zero."""
    front, rear, iso = _tandem_pair(dx=100 * B, dz=0.0)
    tr = solve_tandem(front, rear)
    assert tr.rear.CL < 0.95 * iso.CL         # still strongly coupled


# ------------------------------------------------------ 6. Tier B objective

def test_tandem_problem_dims_and_trim():
    prob = TandemProblem()
    assert prob.dim == 10
    assert prob.bounds.shape == (10, 2)
    x0 = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5, 0.0])
    out = evaluate_tandem(x0, prob)
    assert out["feasible"]
    assert out["CL_total"] == pytest.approx(prob.CL_target, abs=1e-9)
    assert 10.0 < out["LoD"] < 40.0
    # even split, no decalage: front carries MORE lift (upwash) than rear
    assert out["lift_share_front"] > 0.5
    assert out["CDi_mut_rear"] > 0.0 > out["CDi_mut_front"]


def test_tandem_objective_penalty_contract():
    prob = TandemProblem()
    x_bad = np.array([1.5, 1.0, 0, 0, 0, 0, 0, 0, 0.5, 0.0])  # taper > 1
    assert objective_tandem(x_bad, prob) == PENALTY
    assert objective_tandem(np.zeros(3), prob) == PENALTY      # wrong shape
    x0 = np.array([0.5, 0.5, -1, -2, -3, -1, -2, -3, 0.6, 1.0])
    v = objective_tandem(x0, prob)
    assert v == PENALTY or np.isfinite(v)
    assert v > PENALTY                                          # feasible here


def test_tandem_decalage_shifts_lift_split():
    """Positive decalage (rear incidence up) moves lift share aft."""
    prob = TandemProblem()
    base = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5, 0.0])
    up = base.copy()
    up[-1] = 2.0
    o1, o2 = evaluate_tandem(base, prob), evaluate_tandem(up, prob)
    assert o2["lift_share_front"] < o1["lift_share_front"]
    assert o2["CL_total"] == pytest.approx(prob.CL_target, abs=1e-9)
