"""Cross-check helper: run the REFERENCE MATLAB solve_llt (extracted verbatim
from ~/Desktop/GDP/tandem_wing/core/lift_drag_strip.m — not a re-typed copy)
on one geometry and return its results for comparison with the numpy port.

Used by tests/test_llt_elliptic.py::test_cross_check_vs_matlab_solve_llt
(opt-in: AEROBO_MATLAB=1, since a matlab -batch launch costs ~20 s).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

MATLAB_REF = Path.home() / "Desktop/GDP/tandem_wing/core/lift_drag_strip.m"


def extract_solve_llt() -> str:
    """Pull the local function solve_llt out of the reference file verbatim."""
    src = MATLAB_REF.read_text()
    start = src.find("function [CL, Gamma_y, Cl_y, alpha_i_y, A_n] = solve_llt")
    if start < 0:
        raise RuntimeError("solve_llt not found in reference file")
    # runs to the next top-level function declaration (the for-loop inside
    # solve_llt closes with a column-0 'end', so an ^end$ regex truncates)
    nxt = src.find("\nfunction ", start)
    block = src[start:nxt if nxt > 0 else len(src)].rstrip()
    if not block.endswith("end"):
        raise RuntimeError("solve_llt extraction did not end on 'end'")
    return block


DRIVER = """
% auto-generated cross-check driver (see matlab_crosscheck.py)
N = {N}; b = {b}; V = 1.0;
S_target = {S}; taper = {taper};
theta = ((1:N)' - 0.5) * pi / N;
y     = -(b/2) * cos(theta);
eta   = abs(2*y/b);
c_root = 2*S_target/(b*(1+taper));
c = c_root * (1 - (1-taper)*eta);
g.theta = theta; g.y = y; g.c = c; g.N = N; g.b = b;
g.a0_y = deg2rad({a0_deg}) * ones(N,1);
a_y = 2*pi*ones(N,1);
alpha_y = deg2rad({alpha_deg}) + deg2rad({twist_root} + ({twist_tip}-{twist_root})*eta);
[CL, Gamma_y, Cl_y, alpha_i_y, A_n] = solve_llt(g, a_y, alpha_y, b, V);
S = trapz(y, c); AR = b^2/S;
n = (1:N)';
CDi = pi*AR*sum(n .* A_n.^2);
e   = A_n(1)^2 / sum(n .* A_n.^2);
out = struct('CL', CL, 'CDi', CDi, 'e', e, 'AR', AR, 'A1', A_n(1));
fid = fopen('{json_out}', 'w'); fprintf(fid, '%s', jsonencode(out)); fclose(fid);
"""


def run_matlab_solve_llt(N=60, b=10.0, S=10.0, taper=0.5, alpha_deg=5.0,
                         twist_root=0.0, twist_tip=-3.0, a0_deg=-2.0) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        json_out = td / "out.json"
        script = DRIVER.format(N=N, b=b, S=S, taper=taper, alpha_deg=alpha_deg,
                               twist_root=twist_root, twist_tip=twist_tip,
                               a0_deg=a0_deg, json_out=json_out)
        (td / "crosscheck.m").write_text(script + "\n\n" + extract_solve_llt() + "\n")
        subprocess.run(
            ["matlab", "-batch", "crosscheck"],
            cwd=td, check=True, capture_output=True, timeout=300,
        )
        return json.loads(json_out.read_text())
