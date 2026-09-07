"""panel2d: the 2-D multi-body panel method, against references it cannot fake.

The gates here are of three kinds, in descending order of how much they prove:

1. AN INDEPENDENT SOLVER ON BIT-IDENTICAL COORDINATES. XFOIL's inviscid
   solve is a different method (linear-vorticity, second order) written by
   someone else, and it is already a dependency of this package. The
   reference numbers below were produced by driving the installed binary on
   the coordinates these tests generate, and the driving script is recorded
   in the docstring of :func:`_xfoil_reference` so any reader can re-derive
   them rather than trust them.
2. AN EXACT ANALYTIC SOLUTION. The Joukowski map. It is a WEAKER gate than
   it looks, and the test says why: the Joukowski trailing edge is a CUSP,
   which is the one geometry a single-gamma Kutta closure resolves slowly.
3. IDENTITIES THE METHOD MUST SATISFY WHATEVER ITS ACCURACY — symmetry,
   far-field decoupling, the residuals it claims to have driven to zero,
   and determinism.

The trailing-edge story is the reason this file exists. A blunt base panel
was admitted when the module was written; it costs 7.6 % of cl at a
0.0025-chord base and the error GROWS with panel count. It is now refused,
and both halves of that — the refusal and the closure that makes a blunt
section flyable — are gated below.
"""

import numpy as np
import pytest

from aerobo import panel2d as p


# --------------------------------------------------------------- geometry


def naca4(code: str, n: int, sharp: bool = True):
    """NACA 4-digit coordinates in the package's loop order.

    TE -> upper -> LE -> lower -> (closing segment). Cosine-spaced in x.
    ``sharp`` swaps the last thickness coefficient -0.1015 -> -0.1036, the
    standard closure that puts yt(1) = 0 exactly; with ``sharp=False`` the
    section keeps the original coefficient's 0.0021 c base, which is what
    every real 4-digit table ships and what this module now refuses.
    """
    m = int(code[0]) / 100.0
    pp = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n // 2 + 1)
    xc = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if sharp else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                  + 0.2843 * xc ** 3 + a4 * xc ** 4)
    yc = np.where(xc < pp,
                  m / max(pp, 1e-9) ** 2 * (2 * pp * xc - xc ** 2),
                  m / max(1 - pp, 1e-9) ** 2 * ((1 - 2 * pp) + 2 * pp * xc - xc ** 2))
    dyc = np.where(xc < pp,
                   2 * m / max(pp, 1e-9) ** 2 * (pp - xc),
                   2 * m / max(1 - pp, 1e-9) ** 2 * (pp - xc))
    th = np.arctan(dyc)
    xu, yu = xc - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = xc + yt * np.sin(th), yc - yt * np.cos(th)
    # The lower surface runs LE -> TE and always ends AT the trailing edge:
    # with the sharp coefficient its last point coincides with the upper
    # surface's first and the loop closes exactly; with the published one the
    # two sit 2 * yt(1) apart in y at the same x, which is a blunt base.
    return (np.concatenate([xu[::-1], xl[1:]]),
            np.concatenate([yu[::-1], yl[1:]]))


def _xfoil_reference():
    """XFOIL's INVISCID cl for the sharp NACA 2412 of :func:`naca4`.

    Recorded 2026-08-22 from the installed binary (/opt/homebrew/bin/xfoil)
    on the coordinates ``naca4("2412", 200, sharp=True)`` written to file,
    driven with::

        plop / g f / (blank)
        load <coords> / s200
        oper / pacc / <polar file> / (blank)
        aseq 0 8 4 / pacc / (blank)
        quit

    A polar accumulation with no VISC command is an inviscid sweep, which is
    why CD comes back exactly 0.00000 in the recorded table — that zero is
    the evidence the run really was inviscid, and it is the reason these
    numbers are a fair reference for an inviscid panel method.

    Re-derive rather than trust: the same three lines of shell reproduce the
    table, and :func:`test_xfoil_live_agreement` runs it when the binary is
    present.
    """
    return {0.0: 0.2595, 4.0: 0.7414, 8.0: 1.2197}


# ------------------------------------------------- 1. independent solver


@pytest.mark.parametrize("alpha_deg", [0.0, 4.0, 8.0])
def test_agrees_with_xfoil_on_identical_coordinates(alpha_deg):
    """The whole method, against a different one, on the same nodes.

    The tolerance is ABSOLUTE and it is measured, not hoped for. At 200
    panels the error is 3.2e-3, 3.3e-3, 3.4e-3 of cl at alpha 0, 4, 8 — very
    nearly a constant OFFSET rather than a slope error, which is the
    signature of a first-order method resolving a rounded leading edge, and
    is why the slope gate below is a separate and tighter test. A relative
    tolerance would read 1.2 % at alpha 0 (where cl is 0.26) and 0.28 % at
    alpha 8 for the same physical error, and would say nothing.
    """
    x, y = naca4("2412", 200)
    sol = p.solve([p.Body(x=x, y=y, name="2412")], np.deg2rad(alpha_deg),
                  c_ref=1.0)
    ref = _xfoil_reference()[alpha_deg]
    assert sol.cl == pytest.approx(ref, abs=6e-3)


def test_lift_slope_matches_xfoil():
    """The SLOPE, not just the level — a constant offset would pass above.

    A method that loses circulation loses it in proportion to the
    circulation, so the slope is where such an error shows up undiluted.
    """
    x, y = naca4("2412", 200)
    ref = _xfoil_reference()
    cl = {a: p.solve([p.Body(x=x, y=y)], np.deg2rad(a), c_ref=1.0).cl
          for a in (0.0, 8.0)}
    slope = (cl[8.0] - cl[0.0]) / np.deg2rad(8.0)
    slope_ref = (ref[8.0] - ref[0.0]) / np.deg2rad(8.0)
    assert slope == pytest.approx(slope_ref, rel=5e-3)
    # ...and it is the physical slope, a little over 2 pi for a section with
    # thickness (thin-aerofoil theory is the thin-section limit from below).
    assert 2.0 * np.pi < slope < 2.0 * np.pi * 1.15


def test_refinement_converges_towards_xfoil():
    """Refining must make it BETTER.

    This is the test the blunt-trailing-edge defect failed: with a base panel
    the error ran 4.9 % -> 6.5 % -> 8.9 % as the panel count went 100 -> 400
    -> 1600, because the base length is set by the section while every other
    panel shrinks around it. Convergence to the right answer, not the size of
    the error at one count, is what separates a discretisation from a bug.
    """
    ref = _xfoil_reference()[4.0]
    err = []
    for n in (100, 400, 1600):
        x, y = naca4("2412", n)
        sol = p.solve([p.Body(x=x, y=y)], np.deg2rad(4.0), c_ref=1.0)
        err.append(abs(sol.cl - ref) / ref)
    assert err[1] < err[0] and err[2] < err[1], f"not converging: {err}"
    # measured 9.8e-3 -> 2.0e-3 -> 2.1e-4, i.e. about first order in the
    # panel count, which is what a constant-strength method is entitled to
    assert err[0] < 2e-2 and err[2] < 1e-3, f"too coarse: {err}"
    assert err[2] < 0.1 * err[0], f"convergence too slow: {err}"


@pytest.mark.slow
def test_xfoil_live_agreement(tmp_path):
    """The recorded reference, re-derived from the binary if it is installed.

    Marked slow because it shells out. Its job is to catch the recorded
    numbers going stale against a different XFOIL build — a recorded constant
    with no tripwire is how a calibration becomes folklore.
    """
    import shutil
    import subprocess

    exe = shutil.which("xfoil")
    if exe is None:
        pytest.skip("xfoil not installed")
    x, y = naca4("2412", 200)
    np.savetxt(tmp_path / "s.dat", np.column_stack([x, y]), fmt="%12.8f")
    pol = tmp_path / "pol.txt"
    script = ("plop\ng f\n\nload s.dat\ns\noper\npacc\npol.txt\n\n"
              "aseq 0 8 4\npacc\n\nquit\n")
    run = subprocess.run([exe], input=script, text=True, cwd=tmp_path,
                         capture_output=True, timeout=120)
    assert pol.exists(), (
        "xfoil is installed but wrote no polar; the recorded reference "
        f"cannot be checked:\n{run.stdout[-2000:]}")
    rows = [ln.split() for ln in pol.read_text().splitlines()
            if ln.strip() and ln.split()[0].replace(".", "").replace("-", "").isdigit()]
    live = {float(r[0]): float(r[1]) for r in rows}
    for a, recorded in _xfoil_reference().items():
        assert live[a] == pytest.approx(recorded, abs=5e-4), (
            f"recorded XFOIL reference at alpha {a} has gone stale: "
            f"{recorded} vs {live[a]} from the installed binary")


# ------------------------------------------------ 2. the trailing edge


def test_a_blunt_trailing_edge_is_refused_and_says_what_to_do():
    """The defect this module shipped with, closed.

    A 4-digit section with its published thickness law has a 0.0021 c base.
    Panelled as one more solid panel it cost 7.6 % of cl. It is refused, and
    the refusal names the cure rather than leaving the caller to guess.
    """
    x, y = naca4("2412", 200, sharp=False)
    reason = p.check_body(x, y)
    assert reason is not None
    assert "blunt" in reason and "close_trailing_edge" in reason
    with pytest.raises(ValueError, match="blunt"):
        p.Body(x=x, y=y)


def test_closing_a_trailing_edge_makes_it_flyable_and_is_idempotent():
    x, y = naca4("2412", 200, sharp=False)
    gap0 = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    assert gap0 > 0.002
    xc, yc = p.close_trailing_edge(x, y)
    assert float(np.hypot(xc[0] - xc[-1], yc[0] - yc[-1])) == pytest.approx(0.0, abs=1e-14)
    assert p.check_body(xc, yc) is None
    # applying it again is exactly a no-op, bit-for-bit
    xd, yd = p.close_trailing_edge(xc, yc)
    assert np.array_equal(xd, xc) and np.array_equal(yd, yc)


def test_closing_leaves_the_leading_edge_and_the_camber_alone():
    """The closure is a stated change to the SECTION; state what it changes.

    Linear-in-x taper: nothing moves at the leading edge, both surfaces move
    by the same amount in opposite senses, so the thickness closes and the
    camber line is untouched to first order.
    """
    x, y = naca4("2412", 400, sharp=False)
    xc, yc = p.close_trailing_edge(x, y)
    i_le = int(np.argmin(x))
    assert xc[i_le] == pytest.approx(x[i_le], abs=1e-12)
    assert yc[i_le] == pytest.approx(y[i_le], abs=1e-12)
    # the camber at mid-chord is unmoved: the two surfaces there shift by
    # equal and opposite amounts
    up = np.argmin(np.abs(x[:i_le] - 0.5))
    lo = i_le + np.argmin(np.abs(x[i_le:] - 0.5))
    cam_before = 0.5 * (y[up] + y[lo])
    cam_after = 0.5 * (yc[up] + yc[lo])
    assert cam_after == pytest.approx(cam_before, abs=1e-6)
    # ...and the thickness there HAS closed by the tapered fraction
    assert abs(yc[up] - yc[lo]) < abs(y[up] - y[lo])


def test_a_sharp_loop_is_returned_unchanged_bit_for_bit():
    x, y = naca4("2412", 200, sharp=True)
    xc, yc = p.close_trailing_edge(x, y)
    assert np.array_equal(xc, x) and np.array_equal(yc, y)


# --------------------------------------------- 3. the exact solution


def test_joukowski_exact_is_approached_but_the_cusp_is_stated():
    """The analytic gate, and an honest statement of its weakness.

    A Joukowski trailing edge is a CUSP: the two surfaces meet tangentially,
    so the panels either side of the Kutta pair are nearly parallel and the
    single-gamma closure is at its worst. Measured here: 3.5 % at 120 panels
    falling to 0.77 % at 960 — real convergence, at a visibly worse rate than
    the 0.23 % -> 0.04 % the same method delivers on a wedge-angled section.
    The tolerance below is loose ON PURPOSE and the reason is this comment;
    the sharp-section XFOIL gates above are the accuracy claim.
    """
    alpha = np.deg2rad(5.0)
    ex = p.joukowski_exact(alpha, eps_over_a=0.10)
    err = []
    for n in (240, 960):
        xy = p.joukowski_coords(n, eps_over_a=0.10)
        sol = p.solve([p.Body(x=xy[:, 0], y=xy[:, 1])], alpha)
        err.append(abs(sol.bodies[0].cl_gamma - ex["cl"]) / abs(ex["cl"]))
    assert err[1] < err[0], f"cusped case not converging: {err}"
    assert err[1] < 0.02


# ------------------------------------------------------- 4. identities


def test_a_symmetric_section_is_antisymmetric_in_alpha():
    """Exact to round-off — an identity, not an approximation."""
    x, y = naca4("0012", 200)
    cl = {a: p.solve([p.Body(x=x, y=y)], np.deg2rad(a), c_ref=1.0).cl
          for a in (-5.0, 0.0, 5.0)}
    assert cl[0.0] == pytest.approx(0.0, abs=1e-12)
    assert cl[5.0] + cl[-5.0] == pytest.approx(0.0, abs=1e-12)


def test_two_bodies_far_apart_recover_their_isolated_answers():
    """The multi-body machinery must not leak.

    100 chords apart the only physical coupling left is each body's far
    field, a point vortex whose induced angle is O(Gamma / 2 pi r) — a few
    parts in a thousand here, and it must appear with the RIGHT SIGN: the
    downstream body sits in the upstream one's downwash and makes LESS lift.
    A test that only checked "close to isolated" would pass on a solver that
    had silently decoupled them.
    """
    x, y = naca4("2412", 200)
    c = 1.0
    iso = p.solve([p.Body(x=x, y=y, name="a")], np.deg2rad(4.0), c_ref=c)
    pair = p.solve([p.Body(x=x, y=y, name="a"),
                    p.Body(x=x + 100.0, y=y, name="b")],
                   np.deg2rad(4.0), c_ref=c)
    cl_iso = iso.bodies[0].cl
    front, back = pair.bodies[0].cl, pair.bodies[1].cl
    assert front == pytest.approx(cl_iso, rel=0.02)
    assert back == pytest.approx(cl_iso, rel=0.02)
    assert back < cl_iso < front, (
        "the downstream body must lose lift to the upstream one's downwash "
        f"and the upstream one gain it: {front} {cl_iso} {back}")


def test_the_residuals_it_claims_to_drive_to_zero_are_zero():
    x, y = naca4("2412", 200)
    sol = p.solve([p.Body(x=x, y=y, name="a"),
                   p.Body(x=x + 3.0, y=y - 0.3, name="b")], np.deg2rad(6.0))
    assert abs(sol.tangency_residual) < 1e-10
    assert abs(sol.kutta_residual) < 1e-10
    for b in sol.bodies:
        assert abs(b.kutta_residual) < 1e-10


def test_pressure_lift_and_circulation_lift_agree():
    """Two independent routes to the same number.

    The surface pressure integral and Kutta-Joukowski on the solved
    circulation share no arithmetic after the linear solve, so agreement is
    evidence the vortex bookkeeping and the Cp are the same solution.
    """
    x, y = naca4("2412", 800)
    sol = p.solve([p.Body(x=x, y=y)], np.deg2rad(4.0), c_ref=1.0)
    assert sol.cl == pytest.approx(sol.bodies[0].cl_gamma, rel=2e-3)


def test_the_wall_image_equals_an_explicitly_mirrored_body():
    """The image is a convenience, not new physics — so it must be provable.

    A rigid wall carries a mirrored SOURCE of the same sign and a mirrored
    VORTEX of the opposite sign. Building that partner by hand and solving a
    two-body problem must give the real body the same answer as switching the
    image on, up to the one difference that is not a bug: the explicit
    partner is a free body whose own circulation is solved, while the image's
    is slaved. They agree because the mirrored geometry makes the slaved
    value the solved one.
    """
    x, y = naca4("2412", 200)
    y = y + 1.2                                    # clear of the wall
    imaged = p.solve([p.Body(x=x, y=y)], 0.0, c_ref=1.0, image=p.Wall(y=0.0))
    mirror = p.solve([p.Body(x=x, y=y),
                      p.Body(x=x[::-1], y=-y[::-1])], 0.0, c_ref=1.0)
    assert imaged.bodies[0].cl == pytest.approx(mirror.bodies[0].cl, rel=1e-6)


def test_ground_proximity_moves_the_answer_the_way_a_wall_must():
    """A wall near a lifting section changes its lift; far away it does not."""
    x, y = naca4("2412", 200)
    free = p.solve([p.Body(x=x, y=y + 40.0)], 0.0, c_ref=1.0,
                   image=p.Wall(y=0.0))
    near = p.solve([p.Body(x=x, y=y + 0.35)], 0.0, c_ref=1.0,
                   image=p.Wall(y=0.0))
    plain = p.solve([p.Body(x=x, y=y)], 0.0, c_ref=1.0)
    assert free.cl == pytest.approx(plain.cl, rel=2e-3)
    assert abs(near.cl - plain.cl) > 20.0 * abs(free.cl - plain.cl)


def test_the_freestream_must_be_parallel_to_the_wall():
    x, y = naca4("2412", 100)
    with pytest.raises(ValueError, match="parallel to the wall"):
        p.solve([p.Body(x=x, y=y + 1.0)], np.deg2rad(4.0), image=p.Wall(y=0.0))


def test_it_is_deterministic():
    x, y = naca4("2412", 200)
    a = p.solve([p.Body(x=x, y=y)], np.deg2rad(4.0), c_ref=1.0)
    b = p.solve([p.Body(x=x, y=y)], np.deg2rad(4.0), c_ref=1.0)
    assert a.cl == b.cl and a.cm == b.cm
    assert np.array_equal(a.bodies[0].cp, b.bodies[0].cp)


# ------------------------------------------------------ 5. the refusals


@pytest.mark.parametrize("bad,match", [
    ("not_closed", "blunt"),
    ("too_few", "fewer than|at least"),
    ("zero_extent", "zero extent"),
    ("nonfinite", "non-finite"),
    ("reversed", "clockwise"),
])
def test_bad_geometry_is_refused_with_a_reason_not_an_exception(bad, match):
    """check_body is the in-contract path: an optimiser hands it rubbish.

    A CST vector out of a search can be self-intersecting or inside out, and
    that is an infeasible DESIGN, not a programming error, so it gets a
    reason. Body() raises the same string for a call site.
    """
    x, y = naca4("2412", 200)
    if bad == "not_closed":
        x, y = naca4("2412", 200, sharp=False)
    elif bad == "too_few":
        x, y = x[:6], y[:6]
    elif bad == "zero_extent":
        x, y = np.zeros(20), np.zeros(20)
    elif bad == "nonfinite":
        y = y.copy()
        y[3] = np.nan
    elif bad == "reversed":
        x, y = x[::-1], y[::-1]
    reason = p.check_body(x, y)
    assert reason is not None
    assert any(m in reason for m in match.split("|")), reason
    with pytest.raises(ValueError):
        p.Body(x=x, y=y)


def test_a_good_body_is_not_refused():
    x, y = naca4("2412", 200)
    assert p.check_body(x, y) is None


# ------------------------- 6. the two-body gate: d'Alembert on the SUM


def _two_element(n: int = 300, deflection_deg: float = -25.0):
    """A main element and a flap, placed the way a slotted section is placed."""
    x, y = naca4("2412", n)
    main = p.Body(x=x, y=y, name="main")
    flap = (p.Body(x=0.35 * x, y=0.35 * y, name="flap")
            .rotated(np.deg2rad(deflection_deg), 0.0, 0.0)
            .translated(0.92, -0.055))
    return main, flap


@pytest.mark.parametrize("alpha_deg", [0.0, 4.0, 8.0])
def test_the_total_drag_of_a_two_body_system_vanishes(alpha_deg):
    """d'Alembert for a MULTI-BODY system, and the gate nothing else gives.

    In plane potential flow each body of a multi-body system may carry a large
    force in the drag direction — the upstream one carries THRUST, the
    downstream one carries drag — but the TOTAL must vanish. Williams' exact
    two-element solution (ARC R&M 3717, config. A) reports exactly that
    structure: main C_D -0.3839 against flap C_D +0.3838, and the report's own
    sentence, "The total drag is zero, which is consistent with the assumption
    of potential flow."

    Why this gate and not another: a code whose SELF-influence is right and
    whose INTER-BODY influence is wrong passes every lift check and passes the
    far-separation check (where the coupling is negligible by construction).

    WHICH such errors it catches, measured by mutation rather than asserted.
    Scaling the off-diagonal influence blocks by 0.7 in ONE direction only —
    breaking reciprocity — fails all four of these tests. Scaling BOTH
    directions by 0.7 does NOT: the cancellation is a momentum statement, and
    a symmetric error leaves the two bodies' contributions still equal and
    opposite. That case is caught instead by
    :func:`test_the_wall_image_equals_an_explicitly_mirrored_body`, where the
    image's slaved circulation has to agree with a freely solved partner's.
    The two gates are therefore complementary and neither is redundant; both
    were confirmed red under their own mutation and green after restoring.

    Measured here at 400 panels: main -0.150, flap +0.149, sum 1.3e-3, i.e.
    0.9 % of the individual magnitudes, converging 3.2e-2 -> 8.6e-3 -> 2.2e-3
    over 100 -> 400 -> 1600 panels (:func:`test_the_two_body_cancellation_converges`).
    """
    sol = p.solve(list(_two_element(400)), np.deg2rad(alpha_deg), c_ref=1.0)
    cds = [b.cd for b in sol.bodies]
    biggest = max(abs(c) for c in cds)
    # the cancellation must be NON-TRIVIAL: each body really does carry drag,
    # or this test would pass on a solver that found no coupling at all
    assert biggest > 0.05, f"nothing to cancel: {cds}"
    assert cds[0] < 0.0 < cds[1], (
        f"the upstream body must carry thrust and the downstream one drag, "
        f"as Williams' exact solution reports: {cds}")
    assert abs(sum(cds)) < 0.02 * biggest, f"total drag does not vanish: {cds}"


def test_the_two_body_cancellation_converges():
    """Refining must drive the residual towards zero, not merely keep it small.

    A fixed discretisation error and a wrong influence coefficient look alike
    at one panel count; they part company under refinement. This is the same
    diagnostic that exposed the blunt-trailing-edge defect, applied to the
    coupling instead of to the closure.
    """
    ratio = []
    for n in (100, 400, 1600):
        sol = p.solve(list(_two_element(n)), np.deg2rad(4.0), c_ref=1.0)
        cds = [b.cd for b in sol.bodies]
        ratio.append(abs(sum(cds)) / max(abs(c) for c in cds))
    assert ratio[1] < ratio[0] and ratio[2] < ratio[1], f"not converging: {ratio}"
    assert ratio[2] < 0.1 * ratio[0], f"converging too slowly: {ratio}"


def test_a_rigid_motion_preserves_the_closure_and_the_kutta_pair():
    """Placing a flap must not silently move where its Kutta condition acts.

    ``_prepare_loop`` distinguishes a loop whose trailing-edge node is
    REPEATED (closing segment = the last SURFACE panel) from one that is not
    (closing segment = a BASE). ``Body.nodes`` has the repeat already dropped,
    so rebuilding a Body from it re-reads the loop as the second kind and the
    Kutta pair moves inboard by one panel — silently at fine panellings, and
    on exactly the code path that places a flap (rotate, then translate).

    Mutation this catches: routing :meth:`Body.rotated` / :meth:`translated`
    back through the constructor instead of :meth:`Body._moved`.
    """
    x, y = naca4("2412", 100)
    body = p.Body(x=x, y=y, name="m")
    moved = body.rotated(np.deg2rad(-25.0)).translated(0.9, -0.05)
    assert moved.kutta == body.kutta
    assert moved.n_panels == body.n_panels
    # panel lengths are preserved exactly by a rigid motion, in order
    assert np.allclose(moved.geom.L, body.geom.L, rtol=0, atol=1e-12)
    # and the answer does not depend on where the body happens to sit
    here = p.solve([body], np.deg2rad(4.0), c_ref=1.0)
    there = p.solve([body.translated(3.0, -1.0)], np.deg2rad(4.0), c_ref=1.0)
    assert there.cl == pytest.approx(here.cl, rel=1e-12)
