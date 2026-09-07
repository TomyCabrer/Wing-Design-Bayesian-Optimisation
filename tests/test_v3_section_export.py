"""The section can LEAVE the shell.

User report, session 63: *"there should be a way to export airfoil data —
especially important for airfoil only."*

Stage 4 exports the aircraft (an STL, an OpenVSP script, the sections the run
flew). An aerofoil-only session never reaches stage 4 — the whole session IS a
section — so its answer could not leave the shell at all except as a
screenshot. And even in a vehicle session the section is a deliverable of its
own, chosen at stage 2 hours before the wing has been searched.

Three files, and the statements that make each of them worth having:

**The shape is the shape that was DRAWN.** ``section_dat_text`` writes
``stages.airfoil.section_outline``, not a second derivation of it — a database
section keeps its own blunt trailing edge, and only a section with no stored
coordinates falls back to the CST refit. The card and the file cannot show two
different aerofoils.

**The polar says which rows the numbers came from.** The cd and cm this shell
quotes at the design lift are interpolated on the pre-stall monotone branch
only, so the CSV carries an ``on_branch`` column. A section that carries its
metrics but not its curves (a ranked row that did not win the screen) produces
no polar file and says WHY, rather than an empty one.

**Nothing is written that the session does not hold.** The names come from
``cad.FILE_SUFFIXES`` — the one table — and the record is built from the
section object stage 3 flies, so a file that leaves the shell cannot disagree
with the numbers on screen.

Not gated here: ``save_section_export`` is deliberately absent from
``test_v3_act_render_fixpoint.CASES``. It writes files, and the fixpoint suite
drives every case it lists against one shared session with no temporary
directory to write into.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


W_UPPER = [0.15, 0.18, 0.16, 0.20]
W_LOWER = [-0.10, -0.12, -0.08, -0.05]


def _polar_block() -> dict:
    return {"alpha_deg": [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            "cl": [0.0, 0.25, 0.5, 0.75, 0.95, 1.02],
            "cd": [0.0060, 0.0058, 0.0061, 0.0068, 0.0080, 0.0110],
            "cm": [-0.05] * 6,
            "branch": [0, 5], "re": 1.0e6, "mach": 0.0,
            "n_converged": 6, "n_requested": 6}


def _library_section(with_polar: bool = True, blunt: bool = True) -> dict:
    """A pick off the library screen, as ``_adopt_row`` records it."""
    from aerobo import airfoil as af

    xy = np.asarray(af.cst_coords(W_UPPER, W_LOWER), dtype=float)
    if blunt:
        # a real database section: the trailing edge is OPEN, and the CST
        # refit would close it (see stages.airfoil.section_outline)
        xy = xy.copy()
        xy[0, 1] += 0.004
        xy[-1, 1] -= 0.004
    return {"name": "TESTFOIL-15", "source": "library", "rank": 1,
            "tc": 0.1234, "ldcr": 55.25, "clmax": 1.41, "cm": 0.048,
            "w_upper": list(W_UPPER), "w_lower": list(W_LOWER),
            "coords": xy.tolist(), "te_gap": 0.008,
            "conditions": {"re": 1.0e6, "mach": 0.0, "cl_design": 0.5,
                           "tc_min": 0.10, "cm_max": 0.08},
            "report": ({"design": {"polar": _polar_block()},
                        "cl_design": 0.5, "tc_min": 0.10, "cm_max": 0.08}
                       if with_polar else None)}


def _session(sec: dict | None = None, mode: str = "pipeline",
             tmp: Path | None = None):
    from gui.v3 import session as ses

    S = ses.make_session("air")
    if mode == "airfoil":
        assert ses.set_mode(S, "airfoil")
    if sec is not None:
        ses.set_section(S, sec, sec.get("source"), surface="main")
    if tmp is not None:
        ses.set_export_dir(S, str(tmp))
    return S


# ------------------------------------------------------------ the payload
def test_all_three_files_come_out_of_a_chosen_section():
    from aerobo import cad
    from gui.v3 import session as ses

    got = ses.section_export_files(_session(_library_section()), "main")
    assert set(got["files"]) == set(ses.SECTION_EXPORT_KINDS)
    assert got["absent"] == {}
    for kind, f in got["files"].items():
        assert f["name"] == cad.export_name(got["stem"], kind)
        assert f["text"].strip()


def test_no_section_means_three_reasons_and_no_files():
    from gui.v3 import session as ses

    got = ses.section_export_files(_session(), "main")
    assert got["files"] == {}
    assert set(got["absent"]) == set(ses.SECTION_EXPORT_KINDS)
    assert all("no section is chosen" in why
               for why in got["absent"].values())


def test_a_row_that_did_not_win_the_screen_has_no_polar_and_says_so():
    """The screening report keeps the full sweep for its WINNER only. That is
    a missing FILE with a reason, not an empty one."""
    from gui.v3 import session as ses

    got = ses.section_export_files(
        _session(_library_section(with_polar=False)), "main")
    assert "section_polar" not in got["files"]
    assert "WINNER" in got["absent"]["section_polar"]
    # …and the other two still come out: a section with no curves still has a
    # shape and a provenance
    assert "section_dat" in got["files"]
    assert "section_json" in got["files"]


def test_the_second_surface_writes_its_own_files():
    from gui.v3 import session as ses

    S = _session(_library_section())
    aft = {**_library_section(), "name": "TESTFOIL-AFT"}
    ses.set_section(S, aft, "library", surface="aft")
    main = ses.section_export_files(S, "main")
    other = ses.section_export_files(S, "aft")
    assert main["stem"] != other["stem"]
    assert set(main["files"]) == set(other["files"])
    for kind in main["files"]:
        assert main["files"][kind]["name"] != other["files"][kind]["name"]


# ------------------------------------------------------------- the shape
def test_the_dat_is_the_outline_the_card_draws_not_a_refit():
    """The failure this rules out: writing the CST projection, which closes a
    trailing edge the aerofoil really has."""
    from gui.v3 import session as ses
    from gui.v3.stages.airfoil import section_outline

    sec = _library_section()
    text = ses.section_dat_text(sec)
    rows = [ln.split() for ln in text.strip().splitlines()[1:]]
    written = np.asarray([[float(a), float(b)] for a, b in rows])
    drawn = np.asarray(section_outline(sec), dtype=float)
    assert written.shape == drawn.shape
    assert np.allclose(written, drawn, atol=6e-8)
    # the blunt trailing edge SURVIVED — the refit would have closed it
    assert abs(written[0, 1] - written[-1, 1]) > 1e-3
    refit = np.asarray(section_outline(
        {"w_upper": W_UPPER, "w_lower": W_LOWER}), dtype=float)
    assert abs(refit[0, 1] - refit[-1, 1]) < 1e-9


def test_a_section_with_only_weights_falls_back_to_its_refit():
    from gui.v3 import session as ses

    sec = {"name": "CST section (optimised)", "source": "optimised",
           "w_upper": W_UPPER, "w_lower": W_LOWER,
           "conditions": {"re": 1e6, "mach": 0.0, "cl_design": 0.5}}
    text = ses.section_dat_text(sec)
    assert text is not None
    assert text.splitlines()[0] == "CST section (optimised)"


def test_a_section_with_neither_writes_nothing():
    from gui.v3 import session as ses

    assert ses.section_dat_text({"name": "nothing"}) is None
    got = ses.section_export_files(
        _session({"name": "nothing", "source": "library"}), "main")
    assert "section_dat" not in got["files"]
    assert "no outline" in got["absent"]["section_dat"]


# ------------------------------------------------------------- the polar
def test_the_polar_csv_marks_the_branch_the_numbers_came_from():
    from gui.v3 import session as ses

    text = ses.section_polar_csv(_library_section())
    lines = text.strip().splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    assert body[0] == "alpha_deg,cl,cd,cm,on_branch"
    rows = [ln.split(",") for ln in body[1:]]
    assert len(rows) == 6
    # branch [0, 5): the last point is PAST the pre-stall branch
    assert [r[-1] for r in rows] == ["1"] * 5 + ["0"]
    assert float(rows[0][0]) == pytest.approx(-2.0)
    assert float(rows[2][1]) == pytest.approx(0.5)
    # the operating point rides in the comments — a polar with no Reynolds
    # number on it is a table nobody can reuse
    assert any("re: 1e+06" in ln for ln in head)
    assert any("cl_design: 0.5" in ln for ln in head)


def test_a_polar_with_no_recorded_branch_says_so_rather_than_claiming_one():
    from gui.v3 import session as ses

    sec = _library_section()
    sec["report"]["design"]["polar"].pop("branch")
    rows = [ln.split(",") for ln in ses.section_polar_csv(sec).strip()
            .splitlines() if not ln.startswith("#")][1:]
    assert {r[-1] for r in rows} == {"-1"}


# ------------------------------------------------------------- the record
def test_the_json_carries_the_point_the_section_was_designed_at():
    from gui.v3 import session as ses

    rec = ses.section_export_record(_session(_library_section()), "main")
    assert rec["section"]["name"] == "TESTFOIL-15"
    assert rec["section"]["source"] == "library"
    assert rec["designed_at"]["re"] == pytest.approx(1.0e6)
    assert rec["designed_at"]["cl_design"] == pytest.approx(0.5)
    assert rec["weights"]["upper"] == W_UPPER
    assert len(rec["coords"]) > 20
    assert rec["polar"]["n_converged"] == 6
    assert rec["shell"]["surface"] == "main"
    # and it survives a round trip, which is what "JSON-safe" has to mean
    assert json.loads(json.dumps(rec))["section"]["tc"] == pytest.approx(
        0.1234)


def test_an_aerofoil_only_session_exports_the_flow_it_stated():
    """A Reynolds number with no fluid, speed or chord beside it is not
    reproducible — and in this mode the flow is the whole of stage 1."""
    from gui.v3 import session as ses

    S = _session(_library_section(), mode="airfoil")
    rec = ses.section_export_record(S, "main")
    assert rec["shell"]["mode"] == "airfoil"
    assert rec["flow"] is not None
    for key in ("rho", "mu", "re", "mach"):
        assert key in rec["flow"]

    # …and a vehicle session does not pretend to have stated one
    assert "flow" not in ses.section_export_record(
        _session(_library_section()), "main")


# -------------------------------------------------------------- to disk
def test_writing_puts_the_three_files_in_the_chosen_folder(tmp_path):
    from gui.v3 import session as ses

    S = _session(_library_section(), tmp=tmp_path / "out")
    paths = ses.write_section_export(S, "main")
    assert sorted(p.name for p in paths) == sorted(
        p.name for p in (tmp_path / "out").iterdir())
    assert len(paths) == 3
    plan = ses.section_export_plan(S, "main")
    assert plan["existing"] == sorted(p.name for p in paths)


def test_the_folder_is_made_by_the_write_and_not_by_typing_a_path(tmp_path):
    from gui.v3 import session as ses

    target = tmp_path / "not_yet"
    S = _session(_library_section(), tmp=target)
    assert not target.exists()
    ses.write_section_export(S, "main")
    assert target.is_dir()


def test_asking_for_a_file_the_section_cannot_produce_raises(tmp_path):
    from gui.v3 import session as ses

    S = _session(_library_section(with_polar=False), tmp=tmp_path)
    with pytest.raises(ValueError, match="WINNER"):
        ses.write_section_export(S, "main", kinds=("section_polar",))


# ---------------------------------------------------------------- the card
def _texts(view) -> list[str]:
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _shell(mode: str = "pipeline"):
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble("air")
    if mode == "airfoil":
        assert ses.set_mode(ctx.S, "airfoil")
    else:
        ctx.act("accept_mission")
    ses.set_section(ctx.S, _library_section(), "library", surface="main")
    return ctx


def test_the_card_is_on_the_section_view_in_both_modes():
    for mode in ("pipeline", "airfoil"):
        ctx = _shell(mode)
        ctx.render("airfoil", "section")
        said = _texts(ctx.views[("airfoil", "section")])
        assert "Export this section" in said, mode
        assert any("aerobo_design_section.dat" in t for t in said), mode


def _fields(ctx):
    return [e for e in ctx.views[("airfoil", "section")].descendants()
            if type(e).__name__ == "Input"]


def test_this_card_does_not_hand_its_captions_to_quasar():
    """Same defect, same card, other stage: a label passed to ``ui.input``
    is floated by Quasar onto the value in a 26 px field, so "save to"
    printed itself over the start of the path (see the CAD card's own test,
    and ``widgets.select_field``'s docstring for the rule)."""
    ctx = _shell("airfoil")
    ctx.render("airfoil", "section")
    fields = _fields(ctx)
    assert fields, "an aerofoil-only session asks folder and name here"
    for f in fields:
        assert f._props.get("label") is None
    said = _texts(ctx.views[("airfoil", "section")])
    assert "save to" in said and "name" in said


def test_typing_a_folder_moves_what_this_card_says_it_writes():
    """The card quotes the files it would write and warns which it would
    REPLACE. Both were drawn in the same container as the fields, once, so
    an aerofoil-only session could type a new folder and go on reading the
    old one back — the same defect as stage 4's CAD card.

    Typed through the widget, which is the only way the card's OWN handler
    is reached (stage 4 registers the action of that name).
    """
    ctx = _shell("airfoil")             # the mode that asks the folder HERE
    ctx.render("airfoil", "section")
    fields = _fields(ctx)
    assert len(fields) == 2, "an aerofoil-only session asks folder and name"
    said = _texts(ctx.views[("airfoil", "section")])
    assert any("aerobo_design_section.dat" in t for t in said)

    fields[1].set_value("moved")        # the NAME field, one keystroke

    said = _texts(ctx.views[("airfoil", "section")])
    assert any("moved_section.dat" in t for t in said), \
        "the card still names the file it would have written before"
    assert not any("aerobo_design_section.dat" in t for t in said)
    # ...and the field it was typed into survived being re-derived
    assert _fields(ctx) == fields, "a keystroke rebuilt the field"


def test_the_section_card_follows_the_folder_stage_four_asks(tmp_path):
    """One question, one place — so in a VEHICLE session this card quotes
    the folder stage 4 asks rather than asking it again. A quote drawn once
    is a quote that can be of the wrong answer, so changing it at stage 4
    owes this view a repaint."""
    ctx = _shell("pipeline")
    ctx.render("airfoil", "section")
    said = _texts(ctx.views[("airfoil", "section")])
    assert not any(str(tmp_path / "chosen") in t for t in said)

    ctx.act("set_export_dir", str(tmp_path / "chosen"))    # stage 4's handler
    ctx.pay_owed()

    said = _texts(ctx.views[("airfoil", "section")])
    assert any(str(tmp_path / "chosen") in t for t in said), \
        "stage 2 still quotes the folder stage 4 no longer writes to"


def test_the_folder_is_asked_where_there_is_no_stage_four_and_quoted_where():
    """One question, one place: stage 4 asks it in a vehicle session, and an
    aerofoil-only session has no stage 4 to ask it."""
    from gui.v3 import session as ses

    vehicle = _shell("pipeline")
    vehicle.render("airfoil", "section")
    said = _texts(vehicle.views[("airfoil", "section")])
    assert any("asked once, on stage 4" in t for t in said)
    assert str(ses.export_dir(vehicle.S)) in said        # quoted, not typed

    only = _shell("airfoil")
    only.render("airfoil", "section")
    said = _texts(only.views[("airfoil", "section")])
    assert not any("stage 4" in t for t in said)
    assert str(ses.export_dir(only.S)) not in said       # it is a FIELD here


def test_the_button_writes_and_the_card_says_where(tmp_path):
    from gui.v3 import session as ses

    ctx = _shell("airfoil")
    ses.set_export_dir(ctx.S, str(tmp_path))
    ctx.act("save_section_export")
    written = sorted(p.name for p in tmp_path.iterdir())
    assert len(written) == 3
    ctx.render("airfoil", "section")
    said = _texts(ctx.views[("airfoil", "section")])
    assert any(str(tmp_path) in t and "wrote" in t for t in said)


def test_the_card_names_what_it_cannot_write():
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ses.set_section(ctx.S, _library_section(with_polar=False), "library",
                    surface="main")
    ctx.render("airfoil", "section")
    said = _texts(ctx.views[("airfoil", "section")])
    assert any("polar (.csv):" in t and "WINNER" in t for t in said)
