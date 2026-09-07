"""The user chooses WHERE a CAD export lands, and OpenVSP opens on it.

Two things were missing from stage 4's export card, and they are the same
thing twice: an export is a FILE somebody opens in another program, so it
needs a place on this machine.

* every button downloaded through the browser, so the file landed somewhere
  the shell could not name — and "open this in OpenVSP" cannot be done to a
  path nobody knows;
* the OpenVSP button handed over a build SCRIPT. Running it was the user's
  problem, and even run, the model it built opened with VSPAERO's own
  defaults: none of the reference quantities the design was scored on.

So the card asks for a folder, writes there, and the OpenVSP button builds
the ``.vsp3`` (native geometry) with this run's own Sref / bref / cref /
x_cg / Re / Mach and its trim attitude written into the model's VSPAERO
settings, then launches the app on it.

The OpenVSP half needs OpenVSP; those tests skip without it and say so.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from aerobo import api, cad, vsp

from gui.v3 import session


# --------------------------------------------------------------- the folder
def test_the_export_folder_is_the_users_answer(tmp_path):
    S = session.make_session("air")
    assert session.export_dir(S) == session.DEFAULT_EXPORT_DIR

    out = session.set_export_dir(S, str(tmp_path / "here"))
    assert Path(out) == tmp_path / "here"
    assert session.export_dir(S) == tmp_path / "here"
    # ...and nothing is created by ANSWERING: a half-typed path may not
    # litter the filesystem
    assert not (tmp_path / "here").exists()


def test_a_relative_folder_is_resolved_against_the_repo_not_the_cwd():
    """Where the shell happens to have been started is not an answer."""
    S = session.make_session("air")
    session.set_export_dir(S, "results/somewhere")
    assert session.export_dir(S) == session.REPO_ROOT / "results" / "somewhere"
    assert session.export_dir(S).is_absolute()


def test_a_tilde_is_a_home_directory():
    S = session.make_session("air")
    session.set_export_dir(S, "~/aerobo_cad")
    assert session.export_dir(S) == Path.home() / "aerobo_cad"


def test_an_empty_folder_falls_back_rather_than_writing_to_the_cwd():
    S = session.make_session("air")
    session.set_export_dir(S, "   ")
    assert session.export_dir(S) == session.DEFAULT_EXPORT_DIR


@pytest.mark.parametrize("typed,expect", [
    ("", "aerobo_design"),
    ("   ", "aerobo_design"),
    ("wing_v3", "wing_v3"),
    # a path-like answer must not escape the folder the user chose
    ("../../etc/passwd", "....etcpasswd"),
    ("my design!", "mydesign"),
])
def test_the_file_name_cannot_escape_the_folder(typed, expect):
    S = session.make_session("air")
    S["export"]["stem"] = typed
    assert session.export_stem(S) == expect
    assert "/" not in session.export_stem(S)


# ------------------------------------------------- what an export is made of
def _flown(name: str = "tail + winglet", flags: dict | None = None):
    """A design report for a real problem at the middle of its box."""
    cfg = api.RunConfig(problem_name=name, flags=flags or {})
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({} if spec.uses_mission else None, cfg.flags, None)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds], dtype=float)
    return cfg, x, api.design_report(cfg, x)


def test_export_payload_is_the_report_the_page_draws():
    """A file that leaves the shell may not disagree with what is on screen,
    so the payload comes off the SAME design report the geometry view uses."""
    cfg, x, rep = _flown()
    S = session.make_session("air")
    assert session.export_payload(S) is None            # no run yet

    S["run"]["record"] = {"best_x": list(x), "config": cfg.to_dict()
                          if hasattr(cfg, "to_dict") else
                          {"problem_name": cfg.problem_name,
                           "flags": cfg.flags},
                          "param_labels": list(rep["param_labels"])}
    S["run"]["report"] = rep
    data = session.export_payload(S)
    assert data is not None
    assert data["geom"] is rep["geometry"]
    assert data["breakdown"] is rep["breakdown"]
    assert list(data["x"]) == list(x)
    assert data["labels"] == list(rep["param_labels"])


def test_the_bundle_lands_in_the_chosen_folder(tmp_path):
    cfg, x, rep = _flown()
    written = cad.export(rep["geometry"], tmp_path / "chosen", stem="d",
                         x_best=x, labels=rep["param_labels"],
                         section=api.flown_section_coords(cfg, x))
    for path in written.values():
        assert Path(path).parent == tmp_path / "chosen"
    assert (tmp_path / "chosen" / "d.stl").exists()
    assert (tmp_path / "chosen" / "d_vsp.py").exists()


# ------------------------------------------------------------- the PHYSICS
def test_the_aero_setup_is_this_designs_own_numbers():
    """The point of writing VSPAERO's references into the model: a model
    opened in OpenVSP is set up for the aeroplane the run scored, at the
    attitude it was trimmed to — read off the report, never restated."""
    _cfg, _x, rep = _flown()
    geom, bd = rep["geometry"], rep["breakdown"]
    setup = vsp.aero_setup(geom, bd)

    assert setup["sref"] == pytest.approx(float(geom["S"]))
    assert setup["bref"] == pytest.approx(float(geom["b"]))
    assert setup["cref"] == pytest.approx(float(bd["mac"]))
    assert setup["x_cg"] == pytest.approx(float(bd["x_cg"]))
    assert setup["alpha_deg"] == pytest.approx(float(bd["alpha_deg"]))
    assert setup["recref"] == pytest.approx(float(bd["Re_mac"]))
    # the alpha SWEEP straddles the trim point rather than starting at it
    assert setup["alpha_span_deg"] > 0.0
    assert setup["alpha_npts"] >= 3
    # the exported model is FULL span (cad.vsp_script mirrors the wing), so
    # VSPAERO's symmetry flag would fly the aeroplane twice
    assert setup["symmetry"] is False


def test_the_setup_survives_a_report_with_nothing_in_it():
    """A view asks this while drawing itself; it may not raise."""
    setup = vsp.aero_setup({}, {})
    assert setup["sref"] > 0.0 and setup["bref"] > 0.0 and setup["cref"] > 0.0


# ------------------------------------------------------------ the discovery
def test_availability_answers_rather_than_raising():
    have = vsp.availability()
    assert set(have) == {"python", "app", "runner"}
    assert have["runner"] is not None            # it is in this repo
    assert have["runner"].name == "vsp_runner.py"


def test_a_missing_interpreter_is_a_result_not_an_exception(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("AEROBO_VSP_PYTHON", str(tmp_path / "nope"))
    assert vsp.vsp_python() is None
    out = vsp.run_job({"vsp3": "x.vsp3"}, tmp_path)
    assert "AEROBO_VSP_PYTHON" in out["error"]


def test_a_missing_app_is_a_result_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("AEROBO_VSP_APP", str(tmp_path / "nope"))
    assert vsp.vsp_app() is None
    assert "AEROBO_VSP_APP" in vsp.open_in_app(tmp_path / "x.vsp3")["error"]


def test_the_env_var_wins_over_the_search(tmp_path, monkeypatch):
    app = tmp_path / "vsp"
    app.write_text("#!/bin/sh\n")
    app.chmod(0o755)
    monkeypatch.setenv("AEROBO_VSP_APP", str(app))
    assert vsp.vsp_app() == app


# ------------------------------------------ the round trip, where VSP exists
_HAVE_VSP = vsp.vsp_python() is not None
_SKIP = pytest.mark.skipif(
    not _HAVE_VSP or os.environ.get("AEROBO_SKIP_VSP"),
    reason="no OpenVSP interpreter (AEROBO_VSP_PYTHON) on this machine")


@_SKIP
def test_the_model_opens_with_the_physics_already_in_it(tmp_path):
    """The whole point: build the .vsp3 and read the VSPAERO settings BACK
    out of the saved file. Analysis inputs do not persist — parms do — so
    this is the test that would catch the model opening on VSPAERO's own
    defaults instead of on the design's."""
    cfg, x, rep = _flown()
    out = vsp.build_model(rep["geometry"], tmp_path, stem="m", x_best=x,
                          labels=rep["param_labels"],
                          section=api.flown_section_coords(cfg, x),
                          section_aft=api.flown_section_coords(cfg, x,
                                                               aft=True),
                          breakdown=rep["breakdown"])
    assert out["error"] is None, out["error"]
    assert Path(out["vsp3"]).exists()

    want = vsp.aero_setup(rep["geometry"], rep["breakdown"])
    got = out["setup"]
    assert got and "error" not in got
    assert got["Sref"] == pytest.approx(want["sref"])
    assert got["bref"] == pytest.approx(want["bref"])
    assert got["cref"] == pytest.approx(want["cref"])
    assert got["Xcg"] == pytest.approx(want["x_cg"])
    assert got["ReCref"] == pytest.approx(want["recref"])
    assert got["RefFlag"] == 0                    # manual references, not VSP's
    assert got["Symmetry"] == 0

    # ...and the GEOMETRY is native, not an imported mesh: VSP reports the
    # wing as a wing geom whose own span is the developed one
    names = [g["name"] for g in (out.get("geometry") or [])]
    assert "wing" in names and "tail" in names


@_SKIP
def test_the_built_model_reads_back_with_the_settings_saved(tmp_path):
    """Re-open the file through the bridge — a second process, a fresh
    model — and the settings are still there. Saved, not merely set."""
    cfg, x, rep = _flown()
    out = vsp.build_model(rep["geometry"], tmp_path, stem="m", x_best=x,
                          labels=rep["param_labels"],
                          section=api.flown_section_coords(cfg, x),
                          breakdown=rep["breakdown"])
    assert out["error"] is None, out["error"]
    again = vsp.run_job({"vsp3": "m.vsp3", "geometry": True,
                         "setup": vsp.aero_setup(rep["geometry"],
                                                 rep["breakdown"])},
                        tmp_path)
    assert not again.get("error"), again.get("error")
    assert again["setup"]["Sref"] == pytest.approx(float(rep["geometry"]["S"]))


# ------------------------------------------- the card follows its own worker
def _with_a_flown_run(tmp_path):
    """A V3 session with a completed run and its geometry re-evaluated."""
    import time

    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    cfg = config.build_cfg(ctx.S, seed=0)
    spec = api.PROBLEM_SPECS[cfg.problem_name]
    built = spec.build({} if spec.uses_mission else None, cfg.flags, None)
    x = [0.5 * (lo + hi) for lo, hi in built.bounds]
    rows = [{"x": x, "f": built.evaluate(np.array(x))["score"], "g": [0.5],
             "feasible": True}]
    ctx.act("set_result", api.partial_result(cfg, rows).to_dict())
    ctx.render("results", "geometry")
    t0 = time.time()
    while ctx.S["run"]["report"] is None and time.time() - t0 < 60:
        time.sleep(0.02)
    assert ctx.S["run"]["report"] is not None
    ctx.act("set_export_dir", str(tmp_path))
    ctx.act("set_export_stem", "carded")
    ctx.render("results", "log")
    return ctx


def _texts(ctx):
    return [(getattr(e, "text", "") or "")
            for e in ctx.views[("results", "log")].descendants()]


def test_saving_writes_where_the_card_says_and_says_where(tmp_path):
    ctx = _with_a_flown_run(tmp_path)
    ctx.act("save_cad_bundle")
    assert (tmp_path / "carded.stl").exists()
    assert (tmp_path / "carded_vsp.py").exists()
    assert any(str(tmp_path) in t for t in _texts(ctx))


def test_the_card_follows_its_own_background_build(tmp_path, monkeypatch):
    """The build is a subprocess on a worker thread, and a worker may not
    touch a nicegui element — so it publishes and the 0.5 s heartbeat
    redraws. Without the version flag the card kept "BUILDING…" up forever.

    Driven through the FAILING path so it needs no OpenVSP: what is under
    test is that the answer reaches the screen, not what the answer is.
    """
    import time

    monkeypatch.setenv("AEROBO_VSP_PYTHON", str(tmp_path / "no-interpreter"))
    ctx = _with_a_flown_run(tmp_path)
    # the card already SAYS the interpreter is missing (that is the disabled
    # button's reason); what is not on screen yet is the BUILD's own answer
    said = "no OpenVSP interpreter at"
    assert not any(said in t for t in _texts(ctx))

    ctx.act("open_in_vsp")
    dirty = False
    t0 = time.time()
    while time.time() - t0 < 60:
        for fn in list(ctx.polls):
            dirty = bool(fn()) or dirty
        if dirty and any(said in t for t in _texts(ctx)):
            break
        time.sleep(0.05)
    assert dirty, "the heartbeat never saw the build finish"
    assert any(said in t for t in _texts(ctx))


# ============================ the NAME is the user's too, and it is one rule
def test_every_exported_file_is_named_by_the_one_table(tmp_path):
    """``cad.export_name`` is the only speller. The stem is a CONTRACT — the
    generated OpenVSP script re-derives ``STEM + "_wing.dat"`` to find its
    input and writes ``STEM + ".vsp3"`` beside itself — so a second spelling
    anywhere is a bundle that writes every file and still cannot rebuild."""
    with pytest.raises(ValueError, match="unknown export kind"):
        cad.export_name("wing", "step")
    assert cad.export_name("wing", "vsp3") == "wing.vsp3"

    cfg, x, rep = _flown()
    written = cad.export(rep["geometry"], tmp_path, stem="named",
                         x_best=x, labels=rep["param_labels"],
                         section=api.flown_section_coords(cfg, x))
    for kind, suffix in cad.FILE_SUFFIXES.items():
        if kind not in written:            # not every design writes them all
            continue
        assert Path(written[kind]).name == cad.export_name("named", kind)
    # ...and the script it wrote looks for its inputs by the SAME suffixes,
    # which is the drift this table exists to stop
    script = Path(written["vsp"]).read_text()
    assert f"STEM = {'named'!r}" in script
    for kind in ("stl", "dat", "dat_aft", "vsp3"):
        assert f'STEM + "{cad.FILE_SUFFIXES[kind]}"' in script, kind


def test_the_fields_hold_what_was_typed_not_what_it_resolves_to():
    """The card redraws itself (the background VSP build publishes through
    the heartbeat), so a field rebuilt from the RESOLVED value replaces what
    is being typed. The resolution is a readout, not the field."""
    S = session.make_session("air")
    session.set_export_dir(S, "~/Desktop/wings")
    S.setdefault("export", {})["stem"] = "wing v2"

    assert session.export_dir_raw(S) == "~/Desktop/wings"
    assert str(session.export_dir(S)).startswith(str(Path.home()))
    assert session.export_stem_raw(S) == "wing v2"
    assert session.export_stem(S) == "wingv2"       # what the FILE gets


def test_a_redraw_does_not_rewrite_the_fields_under_the_cursor(tmp_path):
    """The same assertion one layer out: what the CARD renders into the two
    inputs. Checking the session helpers alone would pass while the card
    went on painting the resolved path back into the field."""
    ctx = _with_a_flown_run(tmp_path)
    ctx.act("set_export_dir", "~/Desktop/wings")
    ctx.act("set_export_stem", "wing v2")
    ctx.render("results", "log")            # the heartbeat's redraw

    values = [getattr(e, "value", None)
              for e in ctx.views[("results", "log")].descendants()]
    assert "~/Desktop/wings" in values
    assert "wing v2" in values
    # ...and the resolved path is still on screen, as a readout
    assert any(str(Path.home()) in t for t in _texts(ctx))


def test_the_card_says_which_files_and_which_it_will_replace(tmp_path):
    ctx = _with_a_flown_run(tmp_path)
    plan = session.export_plan(ctx.S)
    assert plan["names"]["stl"] == tmp_path / "carded.stl"
    assert plan["existing"] == []
    assert any("carded.stl" in t for t in _texts(ctx))

    ctx.act("save_cad", "stl")
    ctx.render("results", "log")
    assert session.export_plan(ctx.S)["existing"] == ["carded.stl"]
    assert any("will be REPLACED" in t for t in _texts(ctx))

    # ...and a name the filesystem would not take is quoted back, not hidden
    ctx.act("set_export_stem", "wing v2!")
    ctx.render("results", "log")
    assert any("is written as" in t and "wingv2" in t for t in _texts(ctx))


def _inputs(ctx):
    return [e for e in ctx.views[("results", "log")].descendants()
            if type(e).__name__ == "Input"]


def test_typing_a_folder_moves_the_text_that_says_where_it_goes(tmp_path):
    """The reported bug. The card asks WHERE and WHAT NAME, then quotes the
    files back — and the quote was drawn once, in the same container as the
    fields, with nothing redrawing it. So typing a new folder left "writes"
    naming the OLD one, while the buttons wrote to the new one: the one
    sentence on the card whose whole job is to say where a file lands was
    the sentence that could be wrong.

    Nothing is rendered here on purpose. A keystroke is the only event.
    """
    ctx = _with_a_flown_run(tmp_path)
    assert any(str(tmp_path / "carded.stl") in t for t in _texts(ctx))

    ctx.act("set_export_dir", str(tmp_path / "elsewhere"))
    ctx.act("set_export_stem", "moved")

    assert any(str(tmp_path / "elsewhere" / "moved.stl") in t
               for t in _texts(ctx)), "the card still names the old folder"
    assert not any("carded.stl" in t for t in _texts(ctx))
    # ...and the buttons agree with the words: this is the pair that drifted
    assert session.export_plan(ctx.S)["names"]["stl"] == (
        tmp_path / "elsewhere" / "moved.stl")
    ctx.act("save_cad", "stl")
    assert (tmp_path / "elsewhere" / "moved.stl").exists()


def test_a_keystroke_does_not_rebuild_the_field_it_was_typed_into(
        tmp_path, monkeypatch):
    """...and the fix for the above may not reintroduce the bug it was
    written around: re-deriving the card must redraw the DERIVED half only.
    Rebuilding a ``ui.input`` from inside its own handler destroys it under
    the cursor, which the user reports as "it won't let me change it"
    (a_typed_number_must_not_rebuild_its_field).

    The same claim for the two other things that redraw this card: a save
    publishes its answer, and the background OpenVSP build publishes through
    the heartbeat every 0.5 s while it runs. The build is driven through its
    FAILING path, for the reason ``test_the_card_follows_its_own_background
    _build`` gives — what is under test is that the redraw leaves the fields
    alone, not what the build answers, and a test suite may not open a
    desktop application on the machine running it.
    """
    import time

    monkeypatch.setenv("AEROBO_VSP_PYTHON", str(tmp_path / "no-interpreter"))
    ctx = _with_a_flown_run(tmp_path)
    fields = _inputs(ctx)
    assert len(fields) == 2, "the card should ask a folder and a name"

    ctx.act("set_export_dir", str(tmp_path / "half"))     # mid-path
    ctx.act("set_export_stem", "wi")                      # mid-name
    assert _inputs(ctx) == fields, "a keystroke rebuilt the field"

    ctx.act("set_export_dir", str(tmp_path))
    ctx.act("set_export_stem", "carded")
    ctx.act("save_cad", "stl")
    assert _inputs(ctx) == fields, "a save rebuilt the field"

    # ...and wait for the build to FINISH, not merely to start. The first
    # heartbeat fires on the "building the OpenVSP model…" line, which
    # `open_in_vsp` publishes on this thread before the worker has done
    # anything — breaking there ends the test with the worker still
    # running, `monkeypatch` puts the real interpreter back underneath it,
    # and the build then succeeds and OPENS OPENVSP on the machine running
    # the suite. So the loop waits for the worker's own answer, exactly as
    # ``test_the_card_follows_its_own_background_build`` does.
    said = "no OpenVSP interpreter at"
    ctx.act("open_in_vsp")
    t0 = time.time()
    while time.time() - t0 < 60:
        for fn in list(ctx.polls):
            fn()
        if any(said in t for t in _texts(ctx)):
            break
        time.sleep(0.05)
    assert any(said in t for t in _texts(ctx)), \
        "the build never published its answer, so the worker may outlive it"
    assert _inputs(ctx) == fields, "the build's heartbeat rebuilt the field"


def test_no_field_hands_its_caption_to_quasar(tmp_path):
    """The reported symptom, and the one no other test could see: the card
    passed "save to" / "file name" to ``ui.input`` as Quasar's own floating
    label, and this shell pins a field at 26 px. There is no room above the
    value for a floated label, so Quasar drew it ON the value: measured in
    the browser, the caption sat at y 407-417 inside an input whose text ran
    401-425, and the first characters of "/Users/…" were unreadable under
    the word "save to".

    Every other field in this shell puts its caption BESIDE the input
    (``widgets.number_field`` / ``select_field`` / ``text_field``, and
    ``select_field``'s docstring states the rule); these four inputs were
    the only ones that did not, which is why the rule was written down and
    still broken.
    """
    ctx = _with_a_flown_run(tmp_path)
    fields = _inputs(ctx)
    assert fields, "the card should ask a folder and a name"
    for f in fields:
        assert f._props.get("label") is None, (
            "a caption handed to Quasar floats onto the value in a 26 px "
            "field — put it beside the input instead")
    # ...and the captions are still ON SCREEN, as their own labels
    said = _texts(ctx)
    assert "save to" in said and "file name" in said


def test_the_replace_warning_appears_when_the_file_appears(tmp_path):
    """"already in that folder and will be REPLACED" is derived state too:
    it was drawn once, so the first save put the file there and the card
    went on promising a clean write until something else redrew it."""
    ctx = _with_a_flown_run(tmp_path)
    assert not any("will be REPLACED" in t for t in _texts(ctx))
    ctx.act("save_cad", "stl")
    assert any("will be REPLACED" in t and "carded.stl" in t
               for t in _texts(ctx))


def test_saving_one_file_writes_that_file_and_refuses_a_kind_it_has_not(
        tmp_path):
    """``save_cad("vsp")`` wrote the aerofoil .dat: the handler branched on
    "stl" and let everything else fall through one ``else``."""
    ctx = _with_a_flown_run(tmp_path)
    ctx.act("save_cad", "vsp")
    built = tmp_path / "carded_vsp.py"
    assert built.exists(), "the OpenVSP build script was not written"
    assert "OpenVSP" in built.read_text()
    assert not (tmp_path / "carded_wing.dat").exists()

    ctx.act("save_cad", "dat")
    assert (tmp_path / "carded_wing.dat").exists()

    ctx.act("save_cad", "readme")           # not a button on this card
    assert not (tmp_path / "carded_README.txt").exists()
    assert any("is not a file this card writes" in t for t in _texts(ctx))
