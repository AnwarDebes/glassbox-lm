"""Tests of the integrity-failure record (scripts/integrity_failure.py; registration revision 4, section 9.1; the
round-3 review's M1): the record lists every integrity event of the campaign (derived, never chosen); it follows the
campaign's summary (a summary older than the log's last integrity stop is refused); the void bars follow the fixed
tables; the decision is mechanical (rerun_required = the logged fix changes no file named in H1-H5; a byte-for-byte
restoration is no change); the record is write-once; check_pins.py lets the one integrity re-run <c>_icr run exactly
when the record requires it. Uses a fake campaign runs/DRYRUN_inttest (created and removed here) and a registration in
a temporary folder.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_integrity_failure.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import check_pins as CP  # noqa: E402
import integrity_failure as IF  # noqa: E402

CAMP = "DRYRUN_inttest"
PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"
STOP_A2 = "2026-10-22T10:00:00Z [a2a3] STOP [identity_changed] run=A2s1_test: e4_run.py for c/A2s1_test failed (exit 2): x"


@pytest.fixture
def setup(tmp_path):
    cdir = E4 / "runs" / CAMP
    assert not cdir.exists(), "leftover test folder"
    cdir.mkdir(parents=True)
    code = tmp_path / "code.py"
    code.write_text("x = 1\n")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "y.py").write_text("y = 2\n")
    reg = tmp_path / "REGISTRATION_E4.md"
    reg.write_text("## 12. Hashes\n\n### H1 code\n\n| spec | sha256 | files |\n|---|---|---|\n"
                   f"| `{code}` | `{CP.sha256_file(code)}` | 1 |\n| `tree:{tree}` | `{CP.tree_hash(tree)[0]}` | 1 |\n\n"
                   "### H6 the smoke outcome\n\n| spec | sha256 | files |\n|---|---|---|\n\n## 13. Next\n")
    frozen = {"campaign": CAMP, "registration": str(reg), "registration_sha256": CP.sha256_file(reg)}
    try:
        yield {"cdir": cdir, "code": code, "tree": tree, "reg": reg, "frozen": frozen, "tmp": tmp_path}
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.rmtree(E4 / "runs" / f"{CAMP}_icr", ignore_errors=True)


def _summary(cdir, checks=None, log=""):
    ck = {k: True for k in CP.IC_NAMES}
    ck.update(checks or {})
    (cdir / "summary.json").write_text(json.dumps({"integrity": {"checks": ck,
                                                                 "stop_events": CP.stop_effects(log)["events"]}}))
    (cdir / "run_e4_gpu.log").write_text(log)


def test_refusals(setup):
    c, fz = setup["cdir"], setup["frozen"]
    rec, d = IF.build_record(CAMP, fz, "a cause", [], [])
    assert rec is None and any("summary.json" in x and "does not exist" in x for x in d)
    _summary(c)                                                     # every check holds, no stop
    rec, d = IF.build_record(CAMP, fz, "a cause", [], [])
    assert rec is None and any("no bar is void" in x for x in d)
    _summary(c, {"IC5": False})
    for camp, f, cause, needle in ((CAMP + "_icr", {**fz, "campaign": CAMP + "_icr"}, "c", "second re-run"),
                                   (CAMP, {**fz, "campaign": "DRYRUN_other"}, "c", "not the frozen campaign"),
                                   (CAMP, fz, "  ", "--cause is empty"),
                                   (CAMP, {**fz, "registration_sha256": "0" * 64}, "c", "differs from the freeze")):
        rec, d = IF.build_record(camp, f, cause, [], [])
        assert rec is None and any(needle in x for x in d), (needle, d)
    # the summary must be newer than the log's last integrity stop
    (c / "run_e4_gpu.log").write_text(STOP_A2 + "\n")
    rec, d = IF.build_record(CAMP, fz, "a cause", [], [])
    assert rec is None and any("run STAGE=summary again first" in x for x in d)


def test_a_failed_check_and_the_mechanical_decision(setup):
    c, fz = setup["cdir"], setup["frozen"]
    _summary(c, {"IC5": False})
    rec, d = IF.build_record(CAMP, fz, "the store was corrupted by a disk error", [], [])
    assert d == [] and rec["void_bars"] == ["E4.2", "E4.3", "E4.5"] and rec["failed_checks"] == ["IC5"]
    assert rec["rerun_required"] is True and "IS run" in rec["decision"] and rec["logged_utc"].endswith("Z")
    other = setup["tmp"] / "notes.txt"                              # a file no H1-H5 spec names: not pinned
    other.write_text("n\n")
    rec, d = IF.build_record(CAMP, fz, "c", [str(other)], [])
    assert d == [] and rec["rerun_required"] is True and rec["fix_changes"][0]["pinned_by"] == []
    rec, d = IF.build_record(CAMP, fz, "c", [str(setup["code"])], [])   # a pinned file (an H1 file spec)
    assert d == [] and rec["rerun_required"] is False and rec["fix_changes_a_pinned_file"] is True
    assert "not shown (integrity failure)" in rec["decision"]
    rec, d = IF.build_record(CAMP, fz, "c", [str(setup["tree"] / "z_new.py")], [])   # under a tree: spec
    assert d == [] and rec["rerun_required"] is False
    _summary(c, {"IC6": False})                                     # IC6 touches E4.3 and E4.5 only
    rec, d = IF.build_record(CAMP, fz, "c", [], [])
    assert d == [] and rec["void_bars"] == ["E4.3", "E4.5"]


def test_restorations(setup):
    c, fz = setup["cdir"], setup["frozen"]
    _summary(c, {"IC1": False})
    rec, d = IF.build_record(CAMP, fz, "a pinned file was edited and is restored", [], [str(setup["code"])])
    assert d == [] and rec["rerun_required"] is True and rec["restored"][0]["pinned_by"]
    setup["code"].write_text("x = 2\n")                             # not restored byte for byte
    rec, d = IF.build_record(CAMP, fz, "c", [], [str(setup["code"])])
    assert rec is None and any("restore it byte for byte" in x for x in d)
    rec, d = IF.build_record(CAMP, fz, "c", [], [str(setup["tmp"] / "unpinned.txt")])
    assert rec is None and any("no H1-H5 spec" in x for x in d)


def test_an_integrity_stop(setup):
    c, fz = setup["cdir"], setup["frozen"]
    log = "2026-10-22T09:00:00Z [a2a3] pins ok (x)\n" + STOP_A2 + "\n"
    _summary(c, {}, log)                                            # every check holds; the stop voids E4.3, E4.5
    rec, d = IF.build_record(CAMP, fz, "the server was restarted with -np 2", [], [])
    assert d == [] and rec["void_bars"] == ["E4.3", "E4.5"] and rec["failed_checks"] == ["identity_changed"]
    assert rec["failed"][0]["line"] == STOP_A2 and rec["rerun_required"] is True


def test_command_line_write_once_and_the_icr_check(setup):
    c, fz = setup["cdir"], setup["frozen"]
    fpath = setup["tmp"] / "E4_frozen_dry.json"
    fpath.write_text(json.dumps(fz))
    _summary(c, {"IC2a": False})
    icr = CAMP + "_icr"
    assert any("integrity_failure.json" in x for x in CP.frozen_differences(setup["reg"], fpath, icr))
    cmd = [PY, "-I", "-B", str(E4 / "scripts/integrity_failure.py"), "--campaign", CAMP, "--cause", "a test cause",
           "--fix-changes", "--frozen", str(fpath)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    out = c / "integrity_failure.json"
    assert out.stat().st_mode & 0o222 == 0 and json.loads(out.read_text())["rerun_required"] is True
    assert CP.frozen_differences(setup["reg"], fpath, icr) == []    # the record requires the re-run: _icr may run
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 1 and "write-once" in p.stdout
    _summary(c, {"IC2a": True})                                     # the evidence no longer holds: refused again
    assert any("does not read IC2a False" in x for x in CP.frozen_differences(setup["reg"], fpath, icr))
    out.chmod(0o644)
    out.unlink()
    _summary(c, {"IC2a": False})
    p = subprocess.run(cmd[:-3] + ["--fix-changes", str(setup["code"]), "--frozen", str(fpath)], capture_output=True,
                       text=True, timeout=300)                      # the fix changes a pinned file
    assert p.returncode == 0 and json.loads(out.read_text())["rerun_required"] is False, p.stdout + p.stderr
    assert any("does not require the re-run" in x for x in CP.frozen_differences(setup["reg"], fpath, icr))
    p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/integrity_failure.py"), "--campaign", "e4reg_20261022_icr",
                        "--cause", "x", "--fix-changes"], capture_output=True, text=True, timeout=120)
    assert p.returncode != 0 and "REFUSING" in (p.stdout + p.stderr)
