"""Differentiable (torch) twin of the L/D objective — the "adjoint" baseline.

Reverse-mode autodiff through the whole chain (design vector -> chord/twist ->
LLT linear solve -> trim -> profile-drag integral -> L/D) IS the discrete
adjoint method: one backward pass yields the exact gradient at a cost
comparable to one extra primal solve, independent of design dimension. The
crossover benchmark charges each (f, grad) pair 2 evaluations accordingly
(vs d+1 for finite differences).

Trim without brentq: with a constant section slope a_lin the LLT is LINEAR in
the root AoA, so CL(alpha) = CL0 + CL_alpha*alpha and the Fourier coefficients
interpolate linearly between two solves (alpha = 0, 1 rad):

    alpha_trim = (CL_target - CL0) / CL_alpha
    A(alpha_trim) = A0 + alpha_trim * (A1 - A0)

which reproduces the numpy/brentq trim exactly (parity-tested to ~1e-8).

The polar's cd table is evaluated with a differentiable piecewise-linear
interpolant (searchsorted + gather); gradients are exact between knots
(kinks at knots, as with any linear table — standard for L-BFGS use).

Failure contract matches objective.py: PENALTY (-100.0) outside bounds /
polar validity, returned as a constant (zero gradient).
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import (
    TAPER_BOUNDS,
    TWIST_ROOT_BOUNDS_DEG,
    TWIST_TIP_BOUNDS_DEG,
)
from .llt import cosine_stations
from .objective import PENALTY, Problem

_DT = torch.double


def _interp1d(xq: torch.Tensor, xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    """Differentiable piecewise-linear interpolation (clamped ends)."""
    idx = torch.searchsorted(xk, xq.detach().clamp(xk[0], xk[-1]))
    idx = idx.clamp(1, xk.numel() - 1)
    x0, x1 = xk[idx - 1], xk[idx]
    y0, y1 = yk[idx - 1], yk[idx]
    w = ((xq - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return y0 + w * (y1 - y0)


class TorchLLTProblem:
    """Precomputed grids/tables for the differentiable objective.

    Mirrors objective.Problem in trim mode for the crossover knot family:
    x = [taper] + k relative twist knots (deg, root fixed at 0).

    ``fix_root=False`` frees the root knot: x = [taper] + (k+1) knots
    (root bounds = geometry.TWIST_ROOT_BOUNDS_DEG, rest = tip bounds).
    With n_knots=1 that is exactly the Tier A trim design vector
    [taper, twist_root_deg, twist_tip_deg] (linear twist = 2-knot interp).
    """

    def __init__(self, n_knots: int, prob: Problem | None = None,
                 fix_root: bool = True):
        self.prob = prob = prob or Problem(mode="trim")
        self.n_knots = n_knots
        self.fix_root = fix_root
        pol = prob.polar

        theta, y = cosine_stations(prob.N, prob.b)
        self.theta = torch.tensor(theta, dtype=_DT)
        self.y = torch.tensor(y, dtype=_DT)
        self.eta = torch.abs(2.0 * self.y / prob.b)
        self.knot_eta = torch.linspace(0.0, 1.0, n_knots + 1, dtype=_DT)

        N = prob.N
        n = torch.arange(1, N + 1, dtype=_DT)
        self.n = n
        self.sin_jt = torch.sin(self.theta[:, None] * n[None, :])
        self.n_sin_jt = self.sin_jt * n[None, :] / torch.sin(self.theta)[:, None]

        self.a_lin = float(pol.a_lin)
        self.alpha_L0 = float(pol.alpha_L0)
        self.alpha_valid = pol.alpha_valid
        # cd table on a fine grid (exact knots for TablePolar; dense sample
        # for AnalyticPolar) — evaluated with the differentiable interpolant.
        if hasattr(pol, "alpha_deg"):
            ag = np.asarray(pol.alpha_deg, dtype=float)
        else:
            ag = np.linspace(*pol.alpha_valid, 200)
        self.cd_alpha = torch.tensor(ag, dtype=_DT)
        self.cd_val = torch.tensor(np.asarray(pol.cd(ag), dtype=float), dtype=_DT)

        if fix_root:
            self.bounds = np.vstack([np.array([TAPER_BOUNDS]),
                                     np.tile((-6.0, 2.0), (n_knots, 1))])
        else:
            self.bounds = np.vstack([np.array([TAPER_BOUNDS]),
                                     np.array([TWIST_ROOT_BOUNDS_DEG]),
                                     np.tile(TWIST_TIP_BOUNDS_DEG, (n_knots, 1))])

    # ---------------- core differentiable pipeline ----------------

    def _knots(self, x: torch.Tensor) -> torch.Tensor:
        if self.fix_root:
            return torch.cat([torch.zeros(1, dtype=_DT), x[1:]])
        return x[1:]

    def _solve(self, c: torch.Tensor, alpha_geo: torch.Tensor):
        """One LLT solve: returns (A, CL, S, AR)."""
        b = self.prob.b
        M = (4.0 * b / (self.a_lin * c))[:, None] * self.sin_jt + self.n_sin_jt
        A = torch.linalg.solve(M, (alpha_geo - self.alpha_L0)[:, None]).squeeze(1)
        S = torch.trapezoid(c, self.y)
        AR = b**2 / S
        CL = torch.pi * AR * A[0]
        return A, CL, S, AR

    def lod(self, x: torch.Tensor) -> torch.Tensor:
        """Trim-mode L/D as a differentiable scalar (no failure checks)."""
        prob = self.prob
        taper = x[0]
        twist = torch.deg2rad(_interp1d(self.eta, self.knot_eta, self._knots(x)))

        c_root = 2.0 * prob.S / (prob.b * (1.0 + taper))
        c = c_root * (1.0 - (1.0 - taper) * self.eta)

        # LLT is linear in alpha: two solves bracket the whole family
        A0, CL0, S, AR = self._solve(c, twist)
        A1, CL1, _, _ = self._solve(c, twist + 1.0)
        alpha = (prob.CL_target - CL0) / (CL1 - CL0)
        A = A0 + alpha * (A1 - A0)

        CL = torch.pi * AR * A[0]
        CDi = torch.pi * AR * torch.sum(self.n * A**2)
        alpha_i = self.n_sin_jt @ A
        alpha_eff_deg = torch.rad2deg(alpha + twist - alpha_i)
        cd_y = _interp1d(alpha_eff_deg, self.cd_alpha, self.cd_val)
        CDp = torch.trapezoid(cd_y * c, self.y) / S
        return CL / (CDi + CDp + prob.cd0_extra)

    # ---------------- benchmark-facing API ----------------

    def _feasible(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.bounds[:, 0] - 1e-12)
                    and np.all(x <= self.bounds[:, 1] + 1e-12))

    def _alpha_eff_ok(self, x: torch.Tensor) -> bool:
        with torch.no_grad():
            prob = self.prob
            twist = torch.deg2rad(_interp1d(self.eta, self.knot_eta, self._knots(x)))
            c_root = 2.0 * prob.S / (prob.b * (1.0 + x[0]))
            c = c_root * (1.0 - (1.0 - x[0]) * self.eta)
            A0, CL0, _, _ = self._solve(c, twist)
            A1, CL1, _, _ = self._solve(c, twist + 1.0)
            alpha = (prob.CL_target - CL0) / (CL1 - CL0)
            A = A0 + alpha * (A1 - A0)
            aeff = torch.rad2deg(alpha + twist - self.n_sin_jt @ A)
            lo, hi = self.alpha_valid
            return bool(aeff.min() >= lo and aeff.max() <= hi)

    def f(self, x: np.ndarray) -> float:
        """Primal objective (MAXIMISE), same contract as objective.objective."""
        if not self._feasible(x):
            return PENALTY
        xt = torch.tensor(x, dtype=_DT)
        if not self._alpha_eff_ok(xt):
            return PENALTY
        with torch.no_grad():
            v = self.lod(xt)
        return float(v) if torch.isfinite(v) else PENALTY

    def f_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """(f, exact gradient) via one primal + one adjoint (backward) pass.
        Gradient is zero in penalty regions (constant objective there)."""
        if not self._feasible(x):
            return PENALTY, np.zeros_like(np.asarray(x, dtype=float))
        xt = torch.tensor(x, dtype=_DT, requires_grad=True)
        if not self._alpha_eff_ok(xt.detach()):
            return PENALTY, np.zeros_like(np.asarray(x, dtype=float))
        v = self.lod(xt)
        if not torch.isfinite(v):
            return PENALTY, np.zeros_like(np.asarray(x, dtype=float))
        v.backward()
        return float(v), xt.grad.detach().numpy().copy()
