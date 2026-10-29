"""Write E4's write-once integrity-failure record runs/<campaign>/integrity_failure.json (registration revision 4,
section 9.1; the round-3 review's M1: what follows an integrity failure is mechanical and fixed before any number).

When: after the campaign's summary (STAGE=summary) shows a void bar, and after the cause is found and logged (LOG.md
and a research-log row). Run:
  scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/integrity_failure.py --campaign e4reg_<yyyymmdd> \
      --cause "<the logged cause>" --fix-changes [<file> ...] [--restored <file> ...]
The record lists EVERY integrity event of the campaign, derived here, never chosen: each check that reads False in
runs/<c>/summary.json (IC1-IC6) and each integrity stop of runs/<c>/run_e4_gpu.log at a registered stage
(check_pins.stop_effects: exit 2 "identity_changed" when a call ends, exit 7 "pins_changed"). The void bars follow
section 9.1's fixed tables (summarize_e4.SCOPE for a check; check_pins.stop_effects for a stop).
The decision (round-3 review M1 (a)): the files the logged fix changes (--fix-changes; none is a valid answer, for
example a server restored to its pinned identity) are compared with every spec of H1-H5 in the frozen registration; a
file that a spec names or lies under (tree:, dist: files of its RECORD, venv: site-packages) is a pinned file.
  rerun_required = no --fix-changes file is pinned.
  True:  the one integrity re-run <c>_icr IS run, in the next GPU slots of the queue, whatever the numbers
         (sections 9.1 and 11.3); its bars replace the void ones.
  False: the void bars read "not shown (integrity failure)"; no re-run runs under this registration (a fix is a new
         registration revision, a new campaign and a raised test-split use count).
--restored names files restored byte for byte to their registered bytes (section 11.6): each must lie under an H1-H5
spec that hashes to its registered literal again; a restoration is not a change.
Refuses (exit 1, nothing written) unless: the record does not exist (write-once: os.O_EXCL, mode 0444); the campaign
is the frozen one (runs/E4_frozen.json; a DRYRUN_ campaign passes its dry record with --frozen), never its _icr (there
is never a second re-run); the registration equals the freeze record; the cause is not empty; summary.json exists and
was written after the log's last integrity stop (its "stop_events" equal the log's); at least one bar is void; every
--restored file checks out. logged_utc is the clock when the record is written.
check_pins.py refuses every stage of <c>_icr unless this record exists, its evidence still holds and rerun_required is
true; summarize_e4.py reads it for <c> (its decision on every void bar).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import check_pins as CP  # noqa: E402


def scope() -> dict:
    from summarize_e4 import SCOPE

    return SCOPE


def h15_specs(reg_text: str) -> list[tuple[str, str, str]]:
    """(table, spec, literal) of every H1-H5 row."""
    out = []
    for name, rows in CP.parse_registration(reg_text).items():
        if name[:2] in ("H1", "H2", "H3", "H4", "H5"):
            out += [(name, s, h) for s, h in rows]
    return out


def _abs(p: str) -> Path:
    q = Path(p)
    return (q if q.is_absolute() else E4 / q).resolve()


def covering_specs(path: str, specs: list) -> list[tuple[str, str, str]]:
    """The H1-H5 specs that name the file or that it lies under."""
    f = _abs(path)
    out = []
    for name, spec, lit in specs:
        if spec.startswith("tree:"):
            if Path(spec[5:]).resolve() in f.parents:
                out.append((name, spec, lit))
        elif spec.startswith("dist:"):
            di = Path(spec[5:]).resolve()
            if di in f.parents:
                out.append((name, spec, lit))
                continue
            try:
                rec = (di / "RECORD").read_text().splitlines()
            except OSError:
                rec = []
            listed = {(di.parent / (x.rsplit(",", 2)[0] if x.count(",") >= 2 else x.split(",")[0])).resolve()
                      for x in rec if x}
            if f in listed:
                out.append((name, spec, lit))
        elif spec.startswith("venv:"):
            if Path(spec[5:]).resolve() in f.parents:
                out.append((name, spec, lit))
        elif _abs(spec) == f:
            out.append((name, spec, lit))
    return out


def build_record(campaign: str, frozen: dict, cause: str, fix_changes: list, restored: list) -> tuple:
    """(record or None, refusals)."""
    diff = []
    if campaign.endswith(CP.ICR):
        diff.append("an integrity re-run has no integrity re-run of its own (there is never a second re-run)")
    if frozen.get("campaign") != campaign:
        diff.append(f"campaign {campaign!r} is not the frozen campaign {frozen.get('campaign')!r}")
    if not (cause or "").strip():
        diff.append("--cause is empty: the cause is found and logged first (section 9.1)")
    reg_path = Path(frozen.get("registration", ""))
    reg_text = reg_path.read_text() if reg_path.is_file() else ""
    if not reg_text:
        diff.append(f"the frozen registration {reg_path} cannot be read")
    elif CP.sha256_file(reg_path) != frozen.get("registration_sha256"):
        diff.append("the registration differs from the freeze record (restore it first, section 11.6)")
    camp = E4 / "runs" / campaign
    sp, lp = camp / "summary.json", camp / "run_e4_gpu.log"
    try:
        sm = json.loads(sp.read_text())
    except (OSError, json.JSONDecodeError):
        sm = None
        diff.append(f"{sp} does not exist: the record follows the campaign's summary (section 9.1)")
    se = CP.stop_effects(lp.read_text() if lp.is_file() else "")
    if sm is not None and ((sm.get("integrity") or {}).get("stop_events") or []) != se["events"]:
        diff.append("summary.json was written before the log's last integrity stop: run STAGE=summary again first")
    specs = h15_specs(reg_text) if reg_text else []
    rest = []
    for r in restored:
        cov = covering_specs(r, specs)
        if not cov:
            diff.append(f"--restored {r}: no H1-H5 spec names or covers it (list it under --fix-changes instead)")
            continue
        for name, spec, lit in cov:
            got, _n = CP.spec_hash(spec)
            if got != lit:
                diff.append(f"--restored {r}: {name} {spec} hashes to {got}, not its registered {lit}: restore it "
                            "byte for byte first (section 11.6)")
        rest.append({"path": str(_abs(r)), "pinned_by": [s for _n, s, _l in cov]})
    if diff:
        return None, diff
    checks = (sm.get("integrity") or {}).get("checks") or {}
    sc = scope()
    failed = [{"kind": "check", "name": k} for k in CP.IC_NAMES if checks.get(k) is False] + se["events"]
    void = set(se["void_bars"])
    for f in failed:
        if f["kind"] == "check":
            void |= set(sc[f["name"]])
    void_l = [b for b in CP.BARS if b in void]
    if not void_l:
        return None, [f"no bar is void in {campaign} (failed checks: none; integrity stops that void a bar: none): "
                      "no record and no re-run (section 9.1)"]
    ev = CP.integrity_evidence_differences(campaign, {"failed": failed})
    if ev:
        return None, ev
    fx = [{"path": str(_abs(p)), "pinned_by": [s for _n, s, _l in covering_specs(p, specs)]} for p in fix_changes]
    pinned = any(x["pinned_by"] for x in fx)
    rerun = not pinned
    decision = ((f"the one integrity re-run {campaign}{CP.ICR} IS run, in the next GPU slots of the queue, "
                 "whatever the numbers (sections 9.1 and 11.3); its bars replace the void ones") if rerun else
                ("the logged fix changes a file named in H1-H5: the void bars read 'not shown (integrity failure)'; "
                 "no re-run under this registration (section 9.1)"))
    rec = {"what": "E4 integrity-failure record (write-once; scripts/integrity_failure.py; registration section 9.1)",
           "campaign": campaign, "failed": failed, "failed_checks": [f["name"] for f in failed],
           "void_bars": void_l, "judged_readings_not_available": se["judged_not_available"],
           "evidence": {"summary_json_sha256": CP.sha256_file(sp), "checks": checks,
                        "run_log_sha256": CP.sha256_file(lp) if lp.is_file() else None},
           "cause": cause.strip(), "fix_changes": fx, "restored": rest, "fix_changes_a_pinned_file": pinned,
           "rerun_required": rerun, "decision": decision, "registration": str(reg_path),
           "registration_sha256": frozen.get("registration_sha256"),
           "tool_sha256": CP.sha256_file(Path(__file__).resolve()),
           "logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return rec, []


def write_once(path: Path, rec: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as fh:
        fh.write((json.dumps(rec, indent=1) + "\n").encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--cause", required=True)
    ap.add_argument("--fix-changes", nargs="*", required=True,
                    help="every file the logged fix changes (give the flag with no file when it changes none)")
    ap.add_argument("--restored", nargs="*", default=[], help="files restored byte for byte to their registered bytes")
    ap.add_argument("--frozen", default=None, help="DRYRUN_ campaigns only: their dry freeze record")
    a = ap.parse_args()
    dry = a.campaign.startswith("DRYRUN_")
    if a.frozen and not dry:
        sys.exit("REFUSING: --frozen is for DRYRUN_ campaigns only (a registered campaign uses runs/E4_frozen.json)")
    if not dry and not re.match(r"^e4reg_\d{8}$", a.campaign):
        sys.exit("REFUSING: --campaign must be the frozen e4reg_<yyyymmdd> (never its _icr)")
    out = E4 / "runs" / a.campaign / "integrity_failure.json"
    if out.exists():
        print(json.dumps({"written": False, "refused": [f"{out} exists: the record is write-once"]}, indent=1))
        sys.exit(1)
    fp = Path(a.frozen) if a.frozen else CP.FROZEN
    try:
        frozen = json.loads(fp.read_text())
    except (OSError, json.JSONDecodeError):
        print(json.dumps({"written": False, "refused": [f"no readable freeze record at {fp}"]}, indent=1))
        sys.exit(1)
    rec, diff = build_record(a.campaign, frozen, a.cause, a.fix_changes, a.restored)
    if diff:
        print(json.dumps({"written": False, "refused": diff}, indent=1))
        sys.exit(1)
    write_once(out, rec)
    print(json.dumps({"written": True, "record": str(out), "record_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      **{k: rec[k] for k in ("failed_checks", "void_bars", "judged_readings_not_available",
                                             "fix_changes_a_pinned_file", "rerun_required", "decision", "logged_utc")}},
                     indent=1))


if __name__ == "__main__":
    main()
