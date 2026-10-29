"""Tests of the write-once freeze record (scripts/freeze_e4.py; revision 3; the decision 4 of
2026-10-21T07:24:12Z; re-review item 3) and of check_pins.py following it: a registered stage is refused before the
freeze, after any edit of the frozen text and for any other campaign; a text restored byte for byte from the frozen
copy passes again (section 11.6). Uses a fake campaign e4reg_00000000 whose smoke folder runs/SMOKE_e4reg_00000000 is
created and removed here (and runs/SMOKE_DRYRUN_freezetest for the dry mode). Also the tool's CPU gate (cpu_ok's rule:
it waits while the cgroup uses more than 3.5 cores, exit 4 after 20 checks) with a fake load command, and its
refusals (--no-cpu-gate or another gate only in the dry mode).
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_freeze.py"""
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
import freeze_e4 as FZ  # noqa: E402

CAMP = "e4reg_00000000"
PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"


SMOKE12 = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11", "13", "14"]       # an extended smoke (revision 4)


def _smoke_files(sd: Path, decision="GO", slot=300.0, cut=("A3t",), tasks=tuple(SMOKE12)):
    (sd / "A0_smoke").mkdir(parents=True, exist_ok=True)
    (sd / "smoke_report.json").write_text(json.dumps({"decision": decision, "budget": {"slot_minutes": slot,
                                                                                       "cut": list(cut)},
                                                      "gates": {"S0_design": {"tasks": list(tasks)}}}))
    (sd / "confirmations.json").write_text("[]")
    (sd / "confirmations_labels.json").write_text(json.dumps({"labeller": "a person", "labels": {}}))
    (sd / "repeat_check.json").write_text("{}")
    (sd / "A0_smoke" / "manifest.json").write_text("{}")


@pytest.fixture
def smoke_dir():
    sd = E4 / "runs" / f"SMOKE_{CAMP}"
    cdir = E4 / "runs" / CAMP
    assert not sd.exists() and not cdir.exists(), "leftover test folders"
    _smoke_files(sd)
    try:
        yield sd
    finally:
        shutil.rmtree(sd, ignore_errors=True)
        shutil.rmtree(cdir, ignore_errors=True)


def _registration(tmp_path, sd, slot="300", cut="A3t", campaign=CAMP, h6_files=None, name="REGISTRATION_E4.md"):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    rows = "".join(f"| `{sd / x}` | `{CP.sha256_file(sd / x)}` | 1 |\n" for x in (h6_files or CP.H6_FILES))
    text = (f"### 11.4 The freeze step\n\n#### F the freeze values\n\n| name | value |\n|---|---|\n"
            f"| `campaign` | `{campaign}` |\n| `slot_minutes` | `{slot}` |\n| `cut` | `{cut}` |\n\n"
            f"## 12. Hashes\n\n### H1 code\n\n| spec | sha256 | files |\n|---|---|---|\n"
            f"| `{f}` | `{CP.sha256_file(f)}` | 1 |\n\n### H6 the smoke outcome\n\n| spec | sha256 | files |\n"
            f"|---|---|---|\n{rows}\n## 13. Next\n")
    reg = tmp_path / name
    reg.write_text(text)
    return reg


def test_parse_and_validate_the_f_table():
    f = FZ.parse_f_table("#### F the freeze values\n\n| name | value |\n|---|---|\n| `campaign` | `e4reg_20261022` |\n"
                         "| `slot_minutes` | `480` |\n| `cut` | `none` |\n\n## 12.\n| `campaign` | `other` |\n")
    assert f == {"campaign": "e4reg_20261022", "slot_minutes": "480", "cut": "none"}
    v, d = FZ.validate_f(f)
    assert d == [] and v == {"campaign": "e4reg_20261022", "slot_minutes": 480.0, "cut": []}
    v, d = FZ.validate_f({"campaign": "e4reg_<yyyymmdd>", "slot_minutes": "<slot>", "cut": "<cut>"})
    assert len(d) == 3                                       # the draft's placeholders never freeze


def test_freeze_writes_once_and_check_pins_follows_it(tmp_path, smoke_dir):
    reg = _registration(tmp_path, smoke_dir)
    out = tmp_path / "E4_frozen.json"
    rec, diff = FZ.freeze(reg, out)
    assert diff == [], diff
    assert rec["registration_sha256"] == CP.sha256_file(reg) and rec["campaign"] == CAMP
    assert rec["slot_minutes"] == 300.0 and rec["cut"] == ["A3t"] and len(rec["h6"]) == 5 and rec["utc"].endswith("Z")
    assert rec["smoke_tasks"] == SMOKE12                     # revision 4: the summary reads the smoke tasks from here
    assert json.loads(out.read_text())["registration"] == str(reg.resolve())
    assert out.stat().st_mode & 0o222 == 0                   # read-only
    copy = FZ.copy_path(out)
    assert copy.name == "E4_frozen_registration.md" and copy.read_bytes() == reg.read_bytes()
    rec2, diff2 = FZ.freeze(reg, out)                        # write-once
    assert rec2 is None and "write-once" in diff2[0]
    assert CP.check(reg, "a1", CAMP, frozen=out)["differences"] == []
    assert CP.check(reg, "summary", CAMP, frozen=out)["differences"] == []
    assert any("not the frozen campaign" in x for x in CP.check(reg, "a1", "e4reg_11111111", frozen=out)["differences"])
    other = _registration(tmp_path, smoke_dir, name="OTHER.md")          # the same text in another file
    assert any("not the frozen one" in x for x in CP.frozen_differences(other, out, CAMP))
    reg.write_text(reg.read_text().replace("## 13. Next", "## 13. Next (edited after the freeze)"))
    assert any("differs from the frozen" in x for x in CP.check(reg, "a1", CAMP, frozen=out)["differences"])
    reg.write_bytes(copy.read_bytes())                       # restored byte for byte (section 11.6)
    assert CP.check(reg, "a1", CAMP, frozen=out)["differences"] == []
    assert CP.frozen_differences(None, out, CAMP) == []      # the record's own registration path by default


def test_freeze_refuses(tmp_path, smoke_dir):
    out = tmp_path / "rec.json"
    for kw, needle in ((dict(campaign="e4_20261022"), "campaign"), (dict(slot="0"), "slot_minutes"),
                       (dict(slot="x"), "slot_minutes"), (dict(cut="A2s2"), "cut")):
        rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir, **kw), out)
        assert rec is None and any(needle in d for d in diff) and not out.exists(), (kw, diff)
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir, cut="none"), out)       # the report says A3t
    assert rec is None and any("cut" in d for d in diff)
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir, slot="200"), out)       # the report says 300
    assert rec is None and any("slot" in d for d in diff)
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir, h6_files=CP.H6_FILES[:4]), out)
    assert rec is None and any("lacks" in d for d in diff)
    reg = _registration(tmp_path, smoke_dir)
    (smoke_dir / "repeat_check.json").write_text('{"changed": 1}')                  # after its hash was written
    rec, diff = FZ.freeze(reg, out)
    assert rec is None and any(d.startswith("H6") and "on disk" in d for d in diff)
    _smoke_files(smoke_dir, decision="PENDING: S8_yes_matcher")
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir), out)
    assert rec is None and any("not GO" in d for d in diff)
    _smoke_files(smoke_dir, tasks=())                          # revision 4: a report that names no smoke tasks
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir), out)
    assert rec is None and any("names no smoke tasks" in d for d in diff)
    _smoke_files(smoke_dir)
    cdir = E4 / "runs" / CAMP
    cdir.mkdir()
    (cdir / "replay.sqlite").write_bytes(b"")
    rec, diff = FZ.freeze(_registration(tmp_path, smoke_dir), out)
    assert rec is None and any("replay.sqlite" in d for d in diff) and not out.exists()
    assert not FZ.copy_path(out).exists()


def test_dry_freeze_and_the_command_line(tmp_path):
    dcamp = "DRYRUN_freezetest"
    sd, cdir = E4 / "runs" / f"SMOKE_{dcamp}", E4 / "runs" / dcamp
    assert not sd.exists() and not cdir.exists()
    try:
        _smoke_files(sd, slot=100.0, cut=())
        f = tmp_path / "code.py"
        f.write_text("y = 2\n")
        reg = tmp_path / "REG.md"
        reg.write_text(f"## 12.\n\n### H1 code\n\n| spec | sha256 | files |\n|---|---|---|\n"
                       f"| `{f}` | `{CP.sha256_file(f)}` | 1 |\n\n### H6 the smoke outcome\n\n| spec | sha256 |\n|---|---|\n")
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", str(reg),
                            "--dry-campaign", dcamp, "--dry-slot", "100", "--no-cpu-gate"], capture_output=True,
                           text=True, timeout=120)
        assert p.returncode == 0, p.stdout + p.stderr
        rec = json.loads((cdir / "E4_frozen_dry.json").read_text())
        assert rec["campaign"] == dcamp and "DRY RUN" in rec["what"] and rec["cut"] == []
        assert rec["smoke_tasks"] == SMOKE12
        assert CP.check(reg, "a0", dcamp, frozen=cdir / "E4_frozen_dry.json")["differences"] == []
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", str(reg),
                            "--dry-campaign", dcamp, "--dry-slot", "100"], capture_output=True, text=True, timeout=120)
        assert p.returncode == 1 and "write-once" in p.stdout
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", str(reg),
                            "--dry-campaign", "e4reg_20261022"], capture_output=True, text=True, timeout=120)
        assert p.returncode != 0 and "DRYRUN_" in (p.stdout + p.stderr)
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/check_pins.py"), "--registration", str(reg), "--stage",
                            "a0", "--campaign", "e4reg_20261022", "--frozen", str(cdir / "E4_frozen_dry.json")],
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "REFUSING" in p.stdout      # a dry record never serves a registered campaign
    finally:
        shutil.rmtree(sd, ignore_errors=True)
        shutil.rmtree(cdir, ignore_errors=True)


def test_cpu_gate_and_its_refusals(tmp_path):
    assert FZ.cores_of("2026-10-21T09:41:21Z cores 5.50 over 10.0 s; throttled 30 of 100 periods") == 5.5
    assert FZ.cores_of("garbage") is None and FZ.cores_of("") is None
    low = [PY, "-I", "-B", "-c", "print('2026-10-21T00:00:00Z cores 1.25 over 10.0 s; throttled 0 of 100 periods')"]
    high = [PY, "-I", "-B", "-c", "print('2026-10-21T00:00:00Z cores 9.75 over 10.0 s; throttled 0 of 100 periods')"]
    ok, seen = FZ.cpu_gate(load_cmd=low, checks=3, sleep_s=0)
    assert ok is True and len(seen) == 1
    ok, seen = FZ.cpu_gate(load_cmd=high, checks=2, sleep_s=0)
    assert ok is False and len(seen) == 2                       # main() exits 4 then, with nothing written
    ok, _ = FZ.cpu_gate(load_cmd=high, checks=1, sleep_s=0, gate=10.0)
    assert ok is True
    reg = tmp_path / "R.md"
    reg.write_text("x\n")
    frozen_before = FZ.FROZEN.read_bytes() if FZ.FROZEN.exists() else None  # E4 is frozen since 2026-10-23
    for extra, needle in ((["--no-cpu-gate"], "--no-cpu-gate"), (["--cpu-gate", "4.0"], "--cpu-gate"),
                          (["--dry-campaign", "DRYRUN_gatetest", "--cpu-gate", "5"], "--cpu-gate")):
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", str(reg), *extra],
                           capture_output=True, text=True, timeout=120)
        assert p.returncode != 0 and "REFUSING" in (p.stdout + p.stderr) and needle in (p.stdout + p.stderr), extra
    frozen_after = FZ.FROZEN.read_bytes() if FZ.FROZEN.exists() else None
    assert not (E4 / "runs" / "DRYRUN_gatetest").exists() and frozen_after == frozen_before
