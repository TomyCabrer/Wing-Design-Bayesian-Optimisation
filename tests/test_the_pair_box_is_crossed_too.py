"""A tandem's span rows are crossed with its area row — on the gate's own terms.

Reported as *"no recommendation for free span … no solution found for tandem"*,
and it was one defect wearing both faces.

``session.clip_size_box`` crosses the span row with the area row so the box's
CORNERS can pass ``sizing.check_ar``. Two rules kept a tandem out of it:

* "A PAIR IS LEFT ALONE", because ``b^2/S_total`` is not either wing's aspect
  ratio. True, and the wrong conclusion: the solvers check each wing at
  ``0.5 * S_total`` (``tandem.py``/``tandemvlm.py``), so the pair has a clip,
  it just is not the single wing's. Measured on the reported session — spans
  3 – 40 m against an area row of 16 – 44 m^2 — the box spanned **AR
  0.41 – 200** per wing and 36 of 48 draws were refused on the aspect ratio
  before their solver ran;
* the area row arrives ZERO WIDTH wherever the area is stated rather than
  searched (the wing-loading mode: S = W / (W/S)), and the "never invert the
  box" guard read that as an inverted row and threw the whole clip away. That
  is the mode the report was about, and it left the span row crossed with
  nothing at all: 6 – 40 m where the gate allows 6 – 20.

The consequence was the same both ways: nothing in the box flew, so nothing
could be measured, so there was no recommendation for the free span — and the
search spent its budget on refusals.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _tandem(planform: str):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.S["wing"]["choices"]["system"] = "tandem"
    session.apply_choices(ctx.S)
    ctx.act("set_planform", planform)
    return ctx


def test_a_pair_span_row_is_clipped_on_half_the_area():
    """The gate's own convention, asserted as arithmetic: every corner of the
    (span, area) rectangle must pass ``b^2 / (0.5 * S)`` inside AR_LIMITS."""
    from aerobo.sizing import AR_LIMITS
    from gui.v3 import config, session

    ctx = _tandem("free")
    eff = config.effective_bounds(ctx.S)
    rows = session.span_rows(ctx.S)
    assert len(rows) == 2
    s_lo, s_hi = eff[session.AREA_ROW][0]
    for row in rows:
        b_lo, b_hi = eff[row][0]
        assert b_lo ** 2 / (0.5 * s_hi) >= AR_LIMITS[0] - 1e-9, row
        assert b_hi ** 2 / (0.5 * s_lo) <= AR_LIMITS[1] + 1e-9, row


def test_a_stated_area_still_crosses_the_span_row():
    """The wing-loading mode: the area is the mission's number, so the row is
    (S, S) — and a zero-width row is a STATED area, not an inverted one."""
    from aerobo.sizing import AR_LIMITS
    from gui.v3 import config, session

    ctx = _tandem("wing_loading")
    S = ctx.S
    area = float(S["mission"]["s_ref_m2"])
    eff = config.effective_bounds(S)
    for row in session.span_rows(S):
        b_lo, b_hi = eff[row][0]
        assert b_hi <= (AR_LIMITS[1] * 0.5 * area) ** 0.5 + 1e-9, row
        assert b_lo ** 2 / (0.5 * area) >= AR_LIMITS[0] - 1e-9, row


def test_the_clip_is_what_makes_the_free_span_recommendable():
    """The report's own two symptoms, together: with the box crossed, draws
    fly, so the measurement has something to bound and both span rows come
    back recommended."""
    from gui.v3 import config, session

    ctx = _tandem("wing_loading")
    got = ctx.act("auto_recommend", True)
    assert got["basis"] == "admissible"
    assert got["n_admissible"] > 0
    eff = config.effective_bounds(ctx.S)
    for row in session.span_rows(ctx.S):
        assert eff[row][1] == "recommended", (row, eff[row])


def test_most_of_the_pair_box_is_no_longer_refused():
    """The number the report is really about: before this, every draw over a
    tandem's own box was refused before its solver."""
    from aerobo import api
    from gui.v3 import config

    ctx = _tandem("wing_loading")
    probe = api.box_refusal_probe(config.build_cfg(ctx.S), n=32)
    assert probe["frac_refused"] < 1.0
    assert probe["n_feasible"] > 0


def test_a_single_wing_is_clipped_exactly_as_before():
    """The pair's share is 0.5 and a single surface's is 1.0 — the one-wing
    arithmetic this function shipped with must not have moved."""
    from aerobo.sizing import AR_LIMITS
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    eff = config.effective_bounds(ctx.S)
    rows = session.span_rows(ctx.S)
    assert len(rows) == 1
    s_lo, s_hi = eff[session.AREA_ROW][0]
    b_lo, b_hi = eff[rows[0]][0]
    # ...to the rounding the clip does INWARDS, so a corner on the gate
    # cannot be rounded back out through it (``session._tidy``)
    want = (AR_LIMITS[1] * s_lo) ** 0.5
    assert want - 1e-3 <= b_hi <= want
    assert b_lo >= (AR_LIMITS[0] * s_hi) ** 0.5 - 1e-9


def test_a_searched_wing_loading_clips_the_floor_and_not_the_ceiling():
    """Where W/S is a design row the area is ``W_total / (W/S)`` and
    ``W_total`` is closed by a fixed point over the wing's own structure — so
    ``W_N / ws`` is a LOWER bound on the area and nothing bounds it above.

    A floor from an under-estimated area is safe (too low, so it excludes
    nothing the gate accepts). A ceiling from one is not: measured on the
    tandem, draws at b = 12.05 m came back at AR 2.39 — an area near 121 m^2
    against the 25.9 m^2 this arithmetic predicts — so a ceiling drawn from
    it would cut long spans the gate takes.
    """
    from aerobo.sizing import AR_LIMITS
    from gui.v3 import config, session
    from gui.v3.app import assemble

    for system, share in (("single", 1.0), ("tandem", 0.5)):
        ctx = assemble("air")
        ctx.act("accept_mission")
        ctx.S["wing"]["choices"]["system"] = system
        session.apply_choices(ctx.S)
        ctx.act("set_planform", "wing_loading_free")
        assert session.loading_is_searched(ctx.S), system

        eff = config.effective_bounds(ctx.S)
        ws_lo, _ws_hi = eff[session.WS_ROW][0]
        area_hi = float(ctx.S["mission"]["W_N"]) / float(ws_lo)
        for row in session.span_rows(ctx.S):
            b_lo, b_hi = eff[row][0]
            want = (AR_LIMITS[0] * share * area_hi) ** 0.5
            assert abs(b_lo - want) < 1e-2, (system, row, b_lo, want)
            # ...and the top of the row is left where the family put it
            assert b_hi >= 40.0 - 1e-6, (system, row, b_hi)


def test_the_stated_wing_is_still_inside_its_own_box():
    """The rule that outranks the clip: a band the shell writes may not
    exclude the design on screen (``hold_the_stated_design``)."""
    from gui.v3 import config, session

    ctx = _tandem("free")
    eff = config.effective_bounds(ctx.S)
    for row in session.span_rows(ctx.S):
        stated = session.nominal_span(ctx.S, row)
        lo, hi = eff[row][0]
        assert lo <= stated <= hi, (row, stated, lo, hi)
