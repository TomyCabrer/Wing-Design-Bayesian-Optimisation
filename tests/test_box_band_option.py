"""The measured BOX band, shipped as a tracked option rather than as prose.

`RESULTS_SESSION47_VERDICTS.md` §E decided this: the band measured over the
2415 design box and used to search the 2412 box — so the search never saw the
population its own normalisation came from — buys `cd` -0.00056 at the design
lift over 42 paired seeds (sign 0.0009 / Wilcoxon 0.0001 / paired t 0.0001,
n80 = 18) at a null plain-J difference. That retires the one open objection to
the box band (it had only ever been measured on the box it then searched), so
it ships as an OPTION. The shipped library band stays the DEFAULT, because
every published number in report §15 and §16 is measured on it.

What these tests pin, as outcomes rather than as restatements:

* the payloads are tracked and loadable WITHOUT re-running the XFOIL probe —
  `results/` is gitignored, and an option nobody can load is not an option;
* a loaded payload reaches the search as the band the run is actually scored
  on, verified by reading the band back off the built objective;
* the box band cannot masquerade as a screen reference, in either direction;
* the DEFAULT has not moved — a run that states no band still gets the library
  one, byte for byte;
* and the measured mechanism holds in the numbers: the ordering of these bands
  by what they charge for cruise L/D is the ordering of their measured drag.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from aerobo import airfoil_select as A
from aerobo import api


def test_the_measured_bands_are_tracked_and_loadable():
    """`results/` is gitignored. A band a user must spend four minutes of
    XFOIL to regenerate is not a shipped option.

    The old form asked the DISK (`path.exists()`) while claiming the TREE. A
    payload sitting untracked in a working copy passes `exists()` and still
    reaches nobody who clones this repo — which is exactly how a written record
    was lost to gitignored `results/` before. Ask git, not the filesystem.

    It was marked `xfail(strict=True)` while both payloads sat untracked —
    another session's uncommitted work, so staging them was not this test's
    call to make. They are tracked now (commit 5ea524f), the gap the marker
    described is closed, and strict is what made closing it say so out loud:
    the marker is removed rather than left to outlive the gap.
    """
    for path in (A.BOX_BAND_PATH, A.BOX_BAND_2415_PATH):
        assert path.exists(), f"{path} is not on disk at all"
        #: `--error-unmatch` exits non-zero for a path git does not track, so
        #: an ignored or merely-written file fails here rather than passing.
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            cwd=str(path.parent), capture_output=True, text=True)
        assert tracked.returncode == 0, (
            f"{path.name} is NOT tracked by git — it exists only in this "
            f"working copy, so a clone cannot load the option "
            f"({tracked.stderr.strip()})")
        assert tracked.stdout.strip(), (
            f"git tracks no path for {path.name}")
        payload = A.load_box_band(path)
        assert set(payload["bounds"]) == set(A.CRITERIA)
        assert payload["n_records"] > 100
        for k, (lo, hi) in payload["bounds"].items():
            assert hi > lo, k


def test_the_held_out_band_is_the_one_measured_on_another_box():
    """The claim rests on the 2415 band specifically: it was measured over a
    box no search in this project uses, which is what makes it held out. Its
    provenance has to say so, or a later reader cannot tell the two apart."""
    held = A.load_box_band(A.BOX_BAND_2415_PATH)
    assert held["anchor"] == "2415"
    assert "2415" in held["sha"]
    assert "DONOR" in held["source"]
    # …and the in-sample one carries no anchor claim it cannot support
    insample = A.load_box_band(A.BOX_BAND_PATH)
    assert "2415" not in insample["sha"]


def test_a_loaded_band_reaches_the_search():
    """A payload that loads but does not arrive is the flag-nobody-reads bug.
    Read the band back off the BUILT objective, not off the payload."""
    payload = A.load_box_band()
    cfg = api.airfoil_run_config(objective="composite",
                                 score_reference=payload,
                                 score_weights="gdp-sweep")
    _name, ref, _w, _cens = api._airfoil_objective(cfg.flags)
    assert ref.sha == payload["sha"]
    for k in A.CRITERIA:
        assert ref.band(k) == pytest.approx(tuple(payload["bounds"][k]))


def test_the_default_has_not_moved():
    """The decision was to ship an option, NOT to change the default. A run
    that states no band gets the shipped library band, so every published
    number in §15/§16 still describes what runs today."""
    cfg = api.airfoil_run_config(objective="composite",
                                 score_weights="gdp-sweep")
    _name, ref, _w, _cens = api._airfoil_objective(cfg.flags)
    lib = A.load_screen_reference()
    assert ref.sha == lib.sha
    for k in A.CRITERIA:
        assert ref.band(k) == pytest.approx(lib.band(k))


def test_a_box_band_cannot_masquerade_as_a_screen_reference():
    """Its sha is deliberately not a screen sha. `load_screen_reference`
    validates provenance, and a band measured over 268 Sobol points of a design
    box passing as a 631-section library screen would silently re-base every
    composite score in the report."""
    with pytest.raises(ValueError):
        A.load_screen_reference(A.BOX_BAND_PATH)
    with pytest.raises(ValueError):
        A.load_screen_reference(A.BOX_BAND_2415_PATH)


def test_a_degenerate_band_is_refused_not_loaded(tmp_path):
    """A zero-width band divides by zero in every exchange rate. It must fail
    at load, where the file is named, rather than at the first evaluation."""
    good = json.loads(A.BOX_BAND_PATH.read_text())
    bad = dict(good, bounds=dict(good["bounds"], ldcr=[50.0, 50.0]))
    p = tmp_path / "degenerate.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="degenerate band"):
        A.load_box_band(p)

    missing = dict(good, bounds={k: v for k, v in good["bounds"].items()
                                 if k != "cm"})
    q = tmp_path / "missing.json"
    q.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="not a band payload"):
        A.load_box_band(q)


def test_the_measured_mechanism_is_in_the_shipped_numbers():
    """The post-hoc mechanism of `RESULTS_SESSION47_VERDICTS.md` §E, pinned in
    the only part of it that is a property of the FILES rather than of the run:
    what each band charges for cruise L/D relative to |Cm|.

    Measured drag against the library band was 2415 -0.00056, box -0.00029,
    4412 -0.00014 (null), stepnorm +0.00069 — and this ratio orders the bands
    the same way. This asserts the ordering that ships, so a re-measured band
    that broke it could not land silently.
    """
    w = A.PRESETS["gdp-sweep"]
    lib = A.load_screen_reference()

    def ratio(ref):
        r = A.exchange_rates(w, ref)
        return r["ldcr"]["per_step"] / r["cm"]["per_step"]

    def as_ref(payload):
        return A.ScoreReference(
            bounds={k: (v[0], v[1]) for k, v in payload["bounds"].items()},
            sha=payload["sha"], n_records=int(payload["n_records"]))

    held = ratio(as_ref(A.load_box_band(A.BOX_BAND_2415_PATH)))
    box = ratio(as_ref(A.load_box_band(A.BOX_BAND_PATH)))
    library = ratio(lib)
    stepnorm = ratio(A.ScoreReference(
        bounds={k: (v[0], v[1])
                for k, v in A.step_normalised_bands(lib, w).items()},
        sha="stepnorm", n_records=0))

    # the measured drag ordering, as prices
    assert held > box > library > stepnorm
    # and the numbers themselves, so a drift is visible and not just an order
    assert held == pytest.approx(3.553, abs=5e-3)
    assert box == pytest.approx(3.221, abs=5e-3)
    assert library == pytest.approx(2.506, abs=5e-3)
    assert stepnorm == pytest.approx(1.750, abs=5e-3)
