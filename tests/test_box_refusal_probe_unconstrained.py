"""``box_refusal_probe`` crashed on the DEFAULT V3 configuration.

It read the margins from ``built.callable(x)`` for every FEASIBLE draw, and it
unpacked the result before asking whether the problem had any margins to read::

    _fx, gx = built.callable(xa)                    # <- unconditional
    if not built.is_constrained or bool(np.all(...)):

An unconstrained family's callable returns a bare float, so the first draw that
came back feasible raised ``TypeError: cannot unpack non-iterable float
object``. Nothing downstream survived it: "Measure the design box" showed the
TypeError instead of a measurement, and every no-solution diagnostic that reads
this probe — including the one the recommended box uses to NAME the gate that
emptied a box — returned nothing at all.

The guard is why a fully-refused box hid it: with no feasible draw the crashing
line is never reached, so the failure needed a box that mostly WORKS. The
default air wing is one — 41 of 64 draws fly it.

``recommend._admissible`` is the same test written the right way round, and was
correct throughout; this is that shape, applied here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402


def _default_cfg():
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    return config.build_cfg(ctx.S)


def test_the_probe_runs_on_an_unconstrained_family():
    cfg = _default_cfg()
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags, cfg.bounds_overrides)
    # the precondition the crash needed: no margins, and draws that DO fly
    assert not built.is_constrained

    got = api.box_refusal_probe(cfg, n=64)
    assert got["n"] == 64
    assert got["n_feasible"] > 0, "a box with no feasible draw cannot see this"
    assert got["n_feasible"] + got["n_refused"] + got["n_infeasible"] == 64
    # an unconstrained family has no margin to be inadmissible against
    assert got["n_infeasible"] == 0


def test_it_still_names_the_gates_on_a_box_the_mission_empties():
    """The other half: the reasons are what a no-solution card quotes."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    assert session.set_size_from_span_ar(ctx.S, 1.0, 8.0)
    session.sync_wing_from_mission(ctx.S)

    got = api.box_refusal_probe(config.build_cfg(ctx.S), n=64)
    assert got["n_feasible"] == 0
    assert got["reasons"], "an empty box with no reason is not a diagnosis"
    assert sum(int(r["n"]) for r in got["reasons"]) == 64
