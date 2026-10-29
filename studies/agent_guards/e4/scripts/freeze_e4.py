"""Write E4's write-once freeze record (revision 3; the decision 4 of 2026-10-21T07:24:12Z; re-review item 3).

Run at the freeze (registration section 11.4), after the smoke, the label check and the slot:
  scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/freeze_e4.py --registration REGISTRATION_E4.md
It writes runs/E4_frozen.json ONCE (os.O_CREAT | os.O_EXCL, so an existing record is never overwritten; mode 0444):
  registration (absolute path), registration_sha256, utc (the clock when the record is written), campaign,
  slot_minutes, cut, smoke_tasks (revision 4: the smoke report's S0 task list, 10 to 20 tasks by the extension rule;
  summarize_e4.py reads it for the reading without the smoke tasks), h6 (the five rows), smoke_report_sha256, the
  check_pins result, freeze_tool_sha256; and, the same
  way, runs/E4_frozen_registration.md: a byte-for-byte copy of the frozen text (its sha256 is the record's), from which
  a registration changed after the freeze can be restored (section 11.6).
It refuses (exit 1, nothing written) unless
  - the record does not exist yet;
  - the registration's F table (section 11.4) names the campaign (e4reg_ and 8 digits), the slot in minutes (> 0) and
    the cut ("none", "A3t" or "A3t A2s2 A2s3");
  - check_pins.check(stage "freeze") finds no difference: every H1-H5 literal matches, and H6 names exactly the five
    files of section 11.4 under runs/SMOKE_<campaign>/, each matching;
  - runs/SMOKE_<campaign>/smoke_report.json says decision GO, and its budget's slot and cut equal the F table's;
  - runs/<campaign>/replay.sqlite does not exist (the registered store starts empty).
Then the freeze is logged in the research log with the record's
sha256. check_pins.py (every registered stage) and summarize_e4.py refuse a registration whose sha256 differs from the
record's, a registration file other than the frozen one, and any campaign but the frozen one (or its one integrity
re-run <campaign>_icr, section 9.1). If a record is wrong, nothing replaces it: the run stops and a decision is logged.
--dry-campaign DRYRUN_<tag> (the end-to-end dry run only): the record goes to runs/DRYRUN_<tag>/E4_frozen_dry.json (its
copy beside it as E4_frozen_registration_dry.md); H6 and
the F table are not read (a dry run cannot fill them in the draft); the slot comes from --dry-slot and must equal the
dry smoke report's; the H1-H5 literals must still match.
The CPU gate (revision 3; the decision 6, cpu_ok before every check_pins): the check hashes about 10 GB, so
the tool first waits, as run_e4_gpu.sh's cpu_ok does, while this cgroup uses more than 3.5 cores (the decision layer's
scripts/cpu_load.py over 10 s; at most 20 checks about a minute apart), then exits 4 with nothing written.
--no-cpu-gate, and a --cpu-gate other than 3.5 (at most 4.5, the user's limit), are refused unless --dry-campaign is
given (the dry run and the unit tests of the dry mode only). A record that exists already is refused before the gate.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4 / "scripts"))

import check_pins as CP  # noqa: E402

FROZEN = CP.FROZEN
F_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*`([^`]*)`\s*\|")
CUTS = {"none": [], "A3t": ["A3t"], "A3t A2s2 A2s3": ["A3t", "A2s2", "A2s3"]}
CAMPAIGN_RE = re.compile(r"^e4reg_\d{8}$")
DLPY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
CPU_LOAD = f"{_REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py"
CPU_GATE, GATE_CHECKS, GATE_SLEEP_S = 3.5, 20, 50


def cores_of(line: str) -> float | None:
    """The cores figure of a cpu_load.py line ("<utc> cores 3.25 over 10.0 s; throttled ..."), or None."""
    parts = (line or "").split()
    try:
        return float(parts[parts.index("cores") + 1])
    except (ValueError, IndexError):
        return None


def cpu_gate(load_cmd: list | None = None, checks: int = GATE_CHECKS, sleep_s: float = GATE_SLEEP_S,
             gate: float = CPU_GATE) -> tuple[bool, list[str]]:
    """Wait while the cgroup uses more than `gate` cores, as cpu_ok does. Returns (ok, the lines read)."""
    cmd = load_cmd or [DLPY, "-I", "-B", CPU_LOAD, "10"]
    seen = []
    for k in range(checks):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})
        line = (p.stdout or "").strip()
        seen.append(line)
        c = cores_of(line)
        if c is not None and c <= gate:
            return True, seen
        if k + 1 < checks:
            time.sleep(sleep_s)
    return False, seen


def parse_f_table(text: str) -> dict:
    """{name: value} from the table under the heading that starts with '#### F' (the freeze values)."""
    out, inside = {}, False
    for line in text.splitlines():
        if line.startswith("#### F"):
            inside = True
            continue
        if inside and line.startswith("#"):
            break
        if inside:
            m = F_ROW.match(line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def validate_f(f: dict) -> tuple[dict, list[str]]:
    diff, v = [], {}
    c = f.get("campaign", "")
    if not CAMPAIGN_RE.match(c):
        diff.append(f"F table: campaign {c!r} is not e4reg_<yyyymmdd> (8 digits)")
    v["campaign"] = c
    try:
        slot = float(f.get("slot_minutes", ""))
        if not slot > 0:
            raise ValueError
        v["slot_minutes"] = slot
    except ValueError:
        diff.append(f"F table: slot_minutes {f.get('slot_minutes')!r} is not a number of minutes > 0")
    cut = f.get("cut", "")
    if cut not in CUTS:
        diff.append(f"F table: cut {cut!r} is not one of {sorted(CUTS)}")
    else:
        v["cut"] = CUTS[cut]
    return v, diff


def smoke_report_differences(sr_path: Path, slot: float | None, cut: list | None) -> tuple[dict | None, list[str]]:
    if not sr_path.is_file():
        return None, [f"{sr_path} does not exist"]
    sr = json.loads(sr_path.read_text())
    diff = []
    if sr.get("decision") != "GO":
        diff.append(f"the smoke report's decision is {sr.get('decision')!r}, not GO")
    bud = sr.get("budget") or {}
    if slot is not None and bud.get("slot_minutes") != slot:
        diff.append(f"the smoke report's slot {bud.get('slot_minutes')!r} differs from the freeze slot {slot!r}")
    if cut is not None and (bud.get("cut") or []) != cut:
        diff.append(f"the smoke report's cut {bud.get('cut')!r} differs from the freeze cut {cut!r}")
    return sr, diff


def copy_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name.replace("E4_frozen", "E4_frozen_registration").replace(".json", ".md"))


def write_once(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def freeze(reg_path: Path, out_path: Path, dry_campaign: str | None = None, dry_slot: float | None = None) -> tuple:
    """Write the record at out_path and the text's copy beside it (never over an existing file).
    Returns (record or None, differences)."""
    reg_path, out_path = Path(reg_path).resolve(), Path(out_path)
    for q in (out_path, copy_path(out_path)):
        if q.exists():
            return None, [f"{q} exists: the freeze record and its copy are write-once and are never replaced"]
    text = reg_path.read_text()
    diff = []
    if dry_campaign:
        campaign, slot = dry_campaign, dry_slot
        sr, d = smoke_report_differences(E4 / "runs" / f"SMOKE_{campaign}" / "smoke_report.json", slot, None)
        diff += d
        cut = ((sr or {}).get("budget") or {}).get("cut") or []
        cp = CP.check(reg_path, "smoke", campaign)           # H1-H5 (a dry run cannot fill H6)
        h6 = []
    else:
        v, d = validate_f(parse_f_table(text))
        diff += d
        campaign, slot, cut = v.get("campaign"), v.get("slot_minutes"), v.get("cut")
        cp = CP.check(reg_path, "freeze", campaign)
        tables = CP.parse_registration(text)
        h6 = [{"spec": s, "sha256": h} for k in tables if k.startswith("H6") for s, h in tables[k]]
        sr = None
        if campaign:
            sr, d = smoke_report_differences(E4 / "runs" / f"SMOKE_{campaign}" / "smoke_report.json", slot, cut)
            diff += d
    smoke_tasks = [str(t) for t in ((((sr or {}).get("gates") or {}).get("S0_design") or {}).get("tasks") or [])]
    if not smoke_tasks:
        diff.append("the smoke report names no smoke tasks (gates.S0_design.tasks)")
    diff += cp["differences"]
    if campaign and (E4 / "runs" / campaign / "replay.sqlite").exists():
        diff.append(f"runs/{campaign}/replay.sqlite exists: the registered store must not exist at the freeze")
    if diff:
        return None, diff
    srp = E4 / "runs" / f"SMOKE_{campaign}" / "smoke_report.json"
    data = reg_path.read_bytes()
    rec = {"what": "E4 freeze record (write-once; scripts/freeze_e4.py)" + (" - DRY RUN, not a registration" if dry_campaign
                                                                           else ""),
           "registration": str(reg_path), "registration_sha256": hashlib.sha256(data).hexdigest(),
           "registration_copy": str(copy_path(out_path)), "campaign": campaign,
           "slot_minutes": slot, "cut": cut, "smoke_tasks": smoke_tasks, "h6": h6,
           "smoke_report_sha256": CP.sha256_file(srp),
           "check_pins": {"stage": cp["stage"], "tables": cp["tables"], "differences": cp["differences"]},
           "freeze_tool_sha256": CP.sha256_file(Path(__file__).resolve()),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_once(copy_path(out_path), data)
    write_once(out_path, (json.dumps(rec, indent=1) + "\n").encode())
    return rec, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--dry-campaign", default=None, help="DRYRUN_<tag>: the end-to-end dry run's record")
    ap.add_argument("--dry-slot", type=float, default=None)
    ap.add_argument("--no-cpu-gate", action="store_true", help="the dry mode's unit tests only (refused otherwise)")
    ap.add_argument("--cpu-gate", type=float, default=CPU_GATE, help="cores; not 3.5 only with --dry-campaign (<= 4.5)")
    a = ap.parse_args()
    if a.dry_campaign:
        if not a.dry_campaign.startswith("DRYRUN_"):
            sys.exit("REFUSING: --dry-campaign must start with DRYRUN_")
        out = E4 / "runs" / a.dry_campaign / "E4_frozen_dry.json"
    else:
        out = FROZEN
    if a.no_cpu_gate and not a.dry_campaign:
        sys.exit("REFUSING: --no-cpu-gate is for the dry mode's unit tests only (--dry-campaign DRYRUN_<tag>)")
    if (a.cpu_gate != CPU_GATE and not a.dry_campaign) or not 0 < a.cpu_gate <= 4.5:
        sys.exit(f"REFUSING: --cpu-gate {a.cpu_gate} (3.5 for a registration; a dry run may set it, at most 4.5)")
    for q in (out, copy_path(out)):
        if q.exists():
            print(json.dumps({"frozen": False, "refused": [f"{q} exists: the freeze record and its copy are write-once "
                                                           "and are never replaced"]}, indent=1))
            sys.exit(1)
    if not a.no_cpu_gate:
        ok, seen = cpu_gate(gate=a.cpu_gate)
        print(json.dumps({"cpu_gate": {"gate_cores": a.cpu_gate, "checks": len(seen), "last": seen[-1] if seen else None,
                                       "ok": ok}}))
        if not ok:
            print(json.dumps({"frozen": False, "refused": [f"the CPU gate: the cgroup stayed above {a.cpu_gate} cores "
                                                           f"for {len(seen)} checks; nothing written"]}, indent=1))
            sys.exit(4)
    if a.dry_campaign:
        out.parent.mkdir(parents=True, exist_ok=True)
    rec, diff = freeze(Path(a.registration), out, a.dry_campaign, a.dry_slot)
    if diff:
        print(json.dumps({"frozen": False, "refused": diff}, indent=1))
        sys.exit(1)
    print(json.dumps({"frozen": True, "record": str(out), "record_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      **{k: rec[k] for k in ("registration", "registration_sha256", "campaign", "slot_minutes", "cut",
                                             "utc")}}, indent=1))


if __name__ == "__main__":
    main()
