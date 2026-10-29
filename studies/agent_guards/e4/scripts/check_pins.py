"""Compare the LITERAL sha256 values written in the registration with the files on disk (review issue 1), and, at every
registered stage, the registration itself with the write-once freeze record (revision 3; the decision 4).

The registration's hash tables are the record: every table under a heading that starts with "### H" (H1 E4 code, H2
frozen pins and data, H3 read-only inputs, H4 tau2 trees, H5 installed packages, interpreters and the server, H6 the
smoke outcome, filled at the freeze) has rows "| `<spec>` | `<64 hex>` |". A spec is
  <path>               one file (absolute, or relative to the E4 folder);
  tree:<dir>           every file under <dir>, except __pycache__ folders and *.pyc files;
  dist:<dist-info dir> every file an installed package's RECORD lists (paths inside site-packages; no *.pyc);
  venv:<site-packages> EVERY dist-info of a venv (revision 3, re-review item 8): one line per dist-info "dist <name-version>
                       <sha256 of its RECORD>", one per top-level *.pth file "pth <name> <sha256>", one per other top-level
                       entry "entry <name> <dir|file>" (an install, upgrade or removal, a new .pth or a module dropped in
                       without a dist-info all change it; an in-place edit that leaves RECORD alone is caught only for the
                       packages pinned file by file with dist:).
A tree, dist or venv hash is sha256 over its sorted lines joined with tabs and newlines ("MISSING" for a listed file
that is gone). The folder is never checked against a manifest it wrote itself: the literals live in the registration
text, which is frozen at the freeze step.
Stages (run_e4_gpu.sh calls this before every stage; any other stage name is refused):
  smoke            all H1-H5 literals must match (H6 is empty before the freeze);
  freeze           (scripts/freeze_e4.py) all H1-H6 literals must match and H6 must name exactly the five files of section
                   11.4 under runs/SMOKE_<campaign>/: smoke_report.json, confirmations.json, confirmations_labels.json,
                   repeat_check.json, A0_smoke/manifest.json;
  a0 a1 rerun train_guard a2a3 judge summary (the REGISTERED stages):
                   as freeze, AND the freeze record (--frozen, default runs/E4_frozen.json) must exist, name this
                   registration file, hold its exact sha256 and name this campaign (or <campaign>_icr, the one
                   integrity re-run of section 9.1); refused before the freeze and on any difference. The _icr
                   campaign (revision 4; round-3 review M1 (c)) also needs runs/<campaign>/integrity_failure.json,
                   written by scripts/integrity_failure.py, for the frozen campaign, with its failures, the cause, the
                   UTC time it was logged and rerun_required true, AND the evidence it names must hold NOW: every named
                   check reads False in runs/<campaign>/summary.json, every named stop ("identity_changed" or
                   "pins_changed") is a '[<stage>] STOP [<name>]' line of runs/<campaign>/run_e4_gpu.log. A record
                   whose fix changes a file named in H1-H5 (rerun_required false) refuses the _icr: its void bars read
                   "not shown (integrity failure)". a0 also: runs/<campaign>/replay.sqlite must not exist unless the
                   campaign's a0 marker exists (the registered store starts empty). DRYRUN_ campaigns skip H6 only
                   (their freeze record is a dry record inside runs/DRYRUN_*/, passed with --frozen).
Exit 0 = every literal matches; 1 = a difference (listed); 2 = a malformed registration or an unknown stage.
Also: --compute <spec> ... prints the table rows for the given specs (how the literals were made); --venv-list <site>
prints the lines a venv: hash is computed over; --stop-effects <run_e4_gpu.log> prints the bars that the log's
integrity stops void (stop_effects(), section 9.1's fixed table; run_e4_gpu.sh's guard: after such a stop only judge
and summary run in that campaign); --attempts <run_e4_gpu.log> --stage <S> prints the failed attempts of stage S so
far (revision 5, section 11.6; stage_attempts(): a 'STOP [client]' (exit 3) or 'STOP [train_guard]' (exit 6) line is
one failed attempt, no other line counts; run_e4_gpu.sh skips a stage after ATTEMPTS_MAX = 3 of them).
Run: .../python -I -B check_pins.py --registration REGISTRATION_E4.md --stage a1 --campaign <c> --out <json>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`?([0-9a-f]{64})`?\s*\|")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _lines_hash(pairs) -> str:
    return hashlib.sha256("".join(f"{r}\t{h}\n" for r, h in sorted(pairs)).encode()).hexdigest()


def tree_hash(d: Path) -> tuple[str, int]:
    pairs = []
    for p in sorted(d.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        pairs.append((str(p.relative_to(d)), sha256_file(p)))
    return _lines_hash(pairs), len(pairs)


def dist_hash(dist_info: Path) -> tuple[str, int]:
    site = dist_info.parent
    pairs = []
    for line in (dist_info / "RECORD").read_text().splitlines():
        rel = line.rsplit(",", 2)[0] if line.count(",") >= 2 else line.split(",")[0]
        if not rel or rel.startswith("..") or rel.endswith(".pyc") or "__pycache__" in rel or rel.endswith("/RECORD"):
            continue
        p = site / rel
        pairs.append((rel, sha256_file(p) if p.is_file() else "MISSING"))
    return _lines_hash(pairs), len(pairs)


def venv_lines(site: Path) -> list[tuple[str, str, str]]:
    """The lines a venv: hash covers (every dist-info's RECORD hash, every top-level .pth, every other entry's name)."""
    out = []
    for p in sorted(Path(site).iterdir(), key=lambda q: q.name):
        nm = p.name
        if nm == "__pycache__":
            continue
        if p.is_dir() and nm.endswith(".dist-info"):
            rec = p / "RECORD"
            out.append(("dist", nm, sha256_file(rec) if rec.is_file() else "MISSING"))
        elif p.is_file() and nm.endswith(".pth"):
            out.append(("pth", nm, sha256_file(p)))
        else:
            out.append(("entry", nm, "dir" if p.is_dir() else "file"))
    return out


def venv_hash(site: Path) -> tuple[str, int]:
    lines = venv_lines(site)
    h = hashlib.sha256("".join("\t".join(x) + "\n" for x in lines).encode()).hexdigest()
    return h, sum(1 for x in lines if x[0] == "dist")


def spec_hash(spec: str) -> tuple[str, int]:
    if spec.startswith("tree:"):
        return tree_hash(Path(spec[5:]))
    if spec.startswith("dist:"):
        return dist_hash(Path(spec[5:]))
    if spec.startswith("venv:"):
        site = Path(spec[5:])
        return venv_hash(site) if site.is_dir() else ("MISSING", 0)
    p = Path(spec)
    p = p if p.is_absolute() else E4 / p
    if not p.is_file():
        return "MISSING", 0
    return sha256_file(p), 1


def parse_registration(text: str) -> dict:
    """{table heading: [(spec, sha256)]} for every '### H...' section."""
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("### "):
            cur = line[4:].strip() if line[4:].startswith("H") else None
            if cur:
                out[cur] = []
            continue
        if line.startswith("## "):
            cur = None
            continue
        if cur:
            m = ROW.match(line)
            if m:
                out[cur].append((m.group(1), m.group(2)))
    return out


SMOKE_STAGES = ("smoke",)
REGISTERED = ("a0", "a1", "rerun", "train_guard", "a2a3", "judge", "summary")
STAGES = SMOKE_STAGES + ("freeze",) + REGISTERED
H6_FILES = ("smoke_report.json", "confirmations.json", "confirmations_labels.json", "repeat_check.json",
            "A0_smoke/manifest.json")
FROZEN = E4 / "runs" / "E4_frozen.json"
ICR = "_icr"
IC_NAMES = ("IC1", "IC2a", "IC2b", "IC3_IC4", "IC5", "IC6")       # summarize_e4.SCOPE's keys
ICR_STOPS = ("identity_changed", "pins_changed")                  # the stops of section 11.6 that follow 9.1
BARS = ("E4.2", "E4.3", "E4.5")
TEST_ARMS = ("A2s1_test", "A2s2_test", "A2s3_test", "A3_test", "A3t_test")
LOG_LINE = re.compile(r"^(\S+) \[([a-z0-9_]+)\] (.*)$")
START_ARM = re.compile(r"^start arm \S+ split \S+ as (\S+?)/(\S+)")
JUDGED = re.compile(r"^judged (\S+) ")
RUN_TAG = re.compile(r"run=(\S+?):? ")


def base_campaign(campaign: str) -> str:
    return campaign[: -len(ICR)] if campaign.endswith(ICR) else campaign


# ------------------------------------------------------- integrity stops (revision 4; section 9.1; round-3 review M1)
def bars_fed_by_run(run: str) -> tuple:
    """The bars a run feeds (section 9.1's fixed table): A0_base, A1_base and their re-runs feed E4.2, E4.3 and E4.5;
    a test arm or its re-run E4.3 and E4.5; a judge call ('judge:<arm>') no bar. An unknown name feeds every bar."""
    n = run.split("/")[-1]
    if n.startswith("judge:"):
        return ()
    core = n[: -len("_rerun")] if n.endswith("_rerun") else n
    if core in ("A0_base", "A1_base"):
        return BARS
    if core in TEST_ARMS:
        return ("E4.3", "E4.5")
    return BARS


def parse_log(text: str) -> list[tuple[str, str, str]]:
    """run_e4_gpu.log as (utc, stage, message) rows ('<utc> [<stage>] <message>')."""
    out = []
    for x in (text or "").splitlines():
        m = LOG_LINE.match(x)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def stop_effects(log_text: str) -> dict:
    """Every integrity stop of a campaign's run_e4_gpu.log at a REGISTERED stage (the smoke's stops are its gate S1),
    with what it voids by section 9.1's table:
      'STOP [identity_changed] run=<n>' (exit 2 when a call ends): the bars run n feeds; a judge call ('judge:<arm>')
        voids no bar, that arm's judged readings are not available;
      'STOP [pins_changed]' at stage S (exit 7): the bars fed by the stage X that ran before S (the last 'pins ok' line
        before the stop): the runs X started; train_guard: E4.3, E4.5; judge: no bar, the judged readings of the arms
        it judged are not available; the smoke (the freeze check came after it) and the summary: nothing.
    Returns {"events": [...], "void_bars": [...], "judged_not_available": [...]}."""
    lines = parse_log(log_text)
    events, void, judged = [], set(), set()
    for i, (utc, st, msg) in enumerate(lines):
        if st not in REGISTERED:
            continue
        name = next((n for n in ICR_STOPS if msg.startswith(f"STOP [{n}]")), None)
        if name is None:
            continue
        ev = {"kind": "stop", "name": name, "stage": st, "line": f"{utc} [{st}] {msg}"}
        v, j = set(), set()
        if name == "identity_changed":
            m = RUN_TAG.search(msg + " ")
            ev["run"] = m.group(1) if m else "?"
            v = set(bars_fed_by_run(ev["run"]))
            if ev["run"].startswith("judge:"):
                j = {ev["run"][len("judge:"):]}
        else:
            k = next((q for q in range(i - 1, -1, -1) if lines[q][2].startswith("pins ok")), None)
            prev = lines[k][1] if k is not None else None
            runs, jd = [], set()
            for _u, st2, msg2 in (lines[k + 1:i] if k is not None else []):
                if st2 != prev:
                    continue
                m = START_ARM.match(msg2)
                if m:
                    runs.append(m.group(2))
                m = JUDGED.match(msg2)
                if m:
                    jd.add(m.group(1))
            ev.update({"previous_stage": prev, "runs_of_previous_stage": runs})
            if prev == "train_guard":
                v = {"E4.3", "E4.5"}
            elif prev in ("a0", "a1", "rerun", "a2a3"):
                for r in runs:
                    v |= set(bars_fed_by_run(r))
            elif prev == "judge":
                j = jd
        ev["void_bars"] = [b for b in BARS if b in v]
        ev["judged_not_available"] = sorted(j)
        events.append(ev)
        void |= v
        judged |= j
    return {"events": events, "void_bars": [b for b in BARS if b in void], "judged_not_available": sorted(judged)}


# ------------------------------------------------ failed attempts (revision 5; section 11.6; round-4 review M1 (b))
ATTEMPTS_MAX = 3                                   # a stage that exits 3 or 6 runs at most 3 times in all (2 resumes)
FAILED_TAGS = ("client", "train_guard")            # the STOP tags of exit 3 and exit 6: the only failed attempts
STOP_LINE = re.compile(r"^STOP \[([a-z_]+)\]")


def attempt_key(stage: str, after_train_guard: bool) -> str:
    """The counter a stage's attempts go to: the stage itself, except that the report step runs 'rerun' twice (rule 9
    for A0 and A1 after a1; for the test arms after a2a3): its lines after the log's first train_guard line count as
    'rerun_test'."""
    return "rerun_test" if stage == "rerun" and after_train_guard else stage


def stage_attempts(log_text: str) -> dict:
    """Every registered stage's failed attempts, read from run_e4_gpu.log (revision 5; section 11.6): a 'STOP [client]'
    line (exit 3) or 'STOP [train_guard]' line (exit 6) of the stage is one failed attempt, whatever its cause. No other
    line counts: not [slot_end] (the operator's stop, exit 8), [server] or [restart_same_identity] (exit 2), [cpu_gate]
    (exit 4), [milp] (exit 5), [refused] or [pins_changed] (exit 7), nor the integrity stop [identity_changed]. A stage
    with ATTEMPTS_MAX failed attempts is final: run_e4_gpu.sh skips it from then on ('SKIP [stage_failed]', exit 0), and
    summarize_e4.py reads every bar with an arm it left incomplete 'not shown (stage failed)'.
    Returns {"failed": {key: n}, "final": [key, ...], "lines": {key: [line, ...]}, "after_train_guard": bool}."""
    failed, lines, after_tg = {}, {}, False
    for utc, st, msg in parse_log(log_text):
        if st not in REGISTERED:
            continue
        if st == "train_guard":
            after_tg = True
        m = STOP_LINE.match(msg)
        if m and m.group(1) in FAILED_TAGS:
            k = attempt_key(st, after_tg)
            failed[k] = failed.get(k, 0) + 1
            lines.setdefault(k, []).append(f"{utc} [{st}] {msg}")
    return {"failed": failed, "final": sorted(k for k, n in failed.items() if n >= ATTEMPTS_MAX), "lines": lines,
            "after_train_guard": after_tg}


def failed_attempts(log_text: str, stage: str) -> int:
    """The failed attempts so far of the stage that starts now (its counter by attempt_key)."""
    sa = stage_attempts(log_text)
    return sa["failed"].get(attempt_key(stage, sa["after_train_guard"]), 0)


def integrity_evidence_differences(campaign: str, rec: dict) -> list[str]:
    """The evidence an integrity record names must hold NOW (revision 4; round-3 review M1 (c)): a named check reads
    False in runs/<campaign>/summary.json; a named stop is, line for line, in runs/<campaign>/run_e4_gpu.log."""
    failed = rec.get("failed") if isinstance(rec, dict) else None
    if not failed or not isinstance(failed, list):
        return ["the integrity record names no failed check and no stop"]
    camp = E4 / "runs" / campaign
    out, sm, log = [], None, None
    for f in failed:
        kind, name = (f or {}).get("kind"), (f or {}).get("name")
        if kind == "check":
            if name not in IC_NAMES:
                out.append(f"unknown integrity check {name!r} (known: {', '.join(IC_NAMES)})")
                continue
            if sm is None:
                try:
                    sm = json.loads((camp / "summary.json").read_text())
                except (OSError, json.JSONDecodeError):
                    sm = {}
            v = ((sm.get("integrity") or {}).get("checks") or {}).get(name, "absent")
            if v is not False:
                out.append(f"runs/{campaign}/summary.json does not read {name} False (it reads {v!r})")
        elif kind == "stop":
            if name not in ICR_STOPS:
                out.append(f"unknown stop {name!r} (known: {', '.join(ICR_STOPS)})")
                continue
            if log is None:
                try:
                    log = (camp / "run_e4_gpu.log").read_text().splitlines()
                except OSError:
                    log = []
            line = f.get("line") or ""
            if f"STOP [{name}]" not in line or f" [{f.get('stage')}] " not in line or line not in log:
                out.append(f"runs/{campaign}/run_e4_gpu.log does not hold the stop line {line!r}")
        else:
            out.append(f"unknown failure kind {kind!r} (check or stop)")
    return out


def icr_differences(base: str) -> list[str]:
    """The one integrity re-run <base>_icr: its record exists, is complete, names this campaign, requires the re-run,
    and the evidence it names holds now (section 9.1)."""
    f = E4 / "runs" / str(base) / "integrity_failure.json"
    try:
        d = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        d = None
    if not (isinstance(d, dict) and d.get("failed") and d.get("cause") and d.get("logged_utc")):
        return [f"the integrity re-run {base}{ICR} needs {f}, written by scripts/integrity_failure.py (every failed "
                "check and integrity stop of the campaign, the logged cause, the UTC time; section 9.1)"]
    out = []
    if d.get("campaign") != base:
        out.append(f"the integrity record names campaign {d.get('campaign')!r}, not {base!r}")
    out += [f"integrity record: {x}" for x in integrity_evidence_differences(str(base), d)]
    if d.get("rerun_required") is not True:
        out.append("the integrity record does not require the re-run (the logged fix changes a file named in H1-H5): "
                   "the void bars read 'not shown (integrity failure)' and no re-run runs under this registration "
                   "(section 9.1)")
    return out


def h6_differences(rows: list, campaign: str | None) -> list[str]:
    """H6 must name exactly the five smoke files of section 11.4 for SMOKE_<campaign> (re-review item 9)."""
    if not rows:
        return ["H6 (the smoke outcome) is empty: the registration is not frozen after the smoke"]
    if not campaign:
        return ["H6 cannot be checked without --campaign"]
    sd = E4 / "runs" / f"SMOKE_{base_campaign(campaign)}"
    want = {str((sd / f).resolve()) for f in H6_FILES}
    got = {str((Path(sp) if Path(sp).is_absolute() else E4 / sp).resolve()) for sp, _ in rows}
    out = []
    if want - got:
        out.append(f"H6 lacks {sorted(want - got)} (section 11.4 names five files)")
    if got - want:
        out.append(f"H6 names files that are not section 11.4's five: {sorted(got - want)}")
    return out


def frozen_differences(reg_path: Path | None, record_path: Path, campaign: str | None) -> list[str]:
    """The registration must be the frozen one, byte for byte, and the campaign the frozen one (or its _icr re-run)."""
    record_path = Path(record_path)
    if not record_path.is_file():
        return [f"not frozen: {record_path} does not exist (a registered stage runs only after the freeze, section 11.4)"]
    try:
        rec = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"the freeze record {record_path} cannot be read ({type(e).__name__})"]
    out = []
    reg_path = Path(rec.get("registration", "")) if reg_path is None else Path(reg_path)
    if Path(rec.get("registration", "")).resolve() != reg_path.resolve():
        out.append(f"the registration {reg_path} is not the frozen one ({rec.get('registration')})")
    got = sha256_file(reg_path) if reg_path.is_file() else "MISSING"
    if got != rec.get("registration_sha256"):
        out.append(f"the registration's sha256 {got} differs from the frozen {rec.get('registration_sha256')}")
    base = rec.get("campaign")
    if campaign not in (base, f"{base}{ICR}"):
        out.append(f"campaign {campaign!r} is not the frozen campaign {base!r} (or its one integrity re-run {base}{ICR})")
    elif campaign == f"{base}{ICR}":
        out += icr_differences(str(base))
    return out


def check(reg_path: Path, stage: str, campaign: str | None, frozen: Path | None = None) -> dict:
    res = {"registration": str(reg_path), "registration_sha256": sha256_file(reg_path), "stage": stage,
           "campaign": campaign, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tables": {},
           "differences": [], "pins_changed": False}
    if stage not in STAGES:
        res["differences"].append(f"unknown stage {stage!r} (known: {', '.join(STAGES)})")
        res["malformed"] = True
        return res
    tables = parse_registration(reg_path.read_text())
    if not any(k.startswith("H1") for k in tables):
        res["differences"].append("no '### H1' table in the registration")
        res["malformed"] = True
        return res
    dry = bool(campaign) and campaign.startswith("DRYRUN_")
    for name, rows in tables.items():
        if name.startswith("H6") and (stage == "smoke" or dry):
            continue
        n_ok = 0
        for spec, want in rows:
            got, nfiles = spec_hash(spec)
            if got == want:
                n_ok += 1
            else:
                res["differences"].append(f"{name}: {spec}: on disk {got}, registered {want}")
                res["pins_changed"] = True     # a pinned file differs from its literal (run_e4_gpu.sh: [pins_changed])
        res["tables"][name] = {"rows": len(rows), "match": n_ok}
    if stage == "freeze" or (stage in REGISTERED and not dry):
        h6 = [k for k in tables if k.startswith("H6")]
        res["differences"] += h6_differences(tables[h6[0]] if h6 else [], campaign)
    if stage in REGISTERED:
        fp = Path(frozen) if frozen else FROZEN
        res["frozen_record"] = str(fp)
        res["frozen_record_sha256"] = sha256_file(fp) if fp.is_file() else None
        fd = frozen_differences(reg_path, fp, campaign)
        res["differences"] += fd
        if any(x.startswith("the registration's sha256") for x in fd):
            res["pins_changed"] = True         # the registration differs from the freeze record
    if stage == "a0" and campaign:
        store = E4 / "runs" / campaign / "replay.sqlite"
        marker = E4 / "runs" / campaign / ".a0_started"
        if store.exists() and not marker.exists():
            res["differences"].append(f"{store} exists before the registered a0 stage started (it must start empty)")
        res["store_exists"], res["a0_marker"] = store.exists(), marker.exists()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default=str(E4 / "REGISTRATION_E4.md"))
    ap.add_argument("--stage", default="smoke")
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frozen", default=None, help="the freeze record (default runs/E4_frozen.json; another path only "
                                                     "for DRYRUN_ campaigns, inside runs/DRYRUN_*/)")
    ap.add_argument("--compute", nargs="*", default=None, help="print '| `spec` | `sha256` |' rows for these specs")
    ap.add_argument("--venv-list", default=None, help="print the lines a venv: hash covers for this site-packages")
    ap.add_argument("--stop-effects", default=None, help="print the bars the integrity stops of this run_e4_gpu.log "
                                                         "void, as a JSON list (run_e4_gpu.sh's integrity guard)")
    ap.add_argument("--attempts", default=None, help="print the failed attempts so far of --stage in this run_e4_gpu.log "
                                                     "(run_e4_gpu.sh's attempt guard; section 11.6)")
    a = ap.parse_args()
    if a.attempts:
        if a.stage not in REGISTERED:
            print(f"REFUSING: --attempts needs a registered --stage ({', '.join(REGISTERED)})")
            sys.exit(2)
        lp = Path(a.attempts)
        print(failed_attempts(lp.read_text() if lp.is_file() else "", a.stage))
        return
    if a.stop_effects:
        lp = Path(a.stop_effects)
        print(json.dumps(stop_effects(lp.read_text() if lp.is_file() else "")["void_bars"]))
        return
    if a.compute is not None:
        for spec in a.compute:
            h, n = spec_hash(spec)
            print(f"| `{spec}` | `{h}` | {n} |")
        return
    if a.venv_list:
        for x in venv_lines(Path(a.venv_list)):
            print("\t".join(x))
        return
    if a.frozen:
        fp = Path(a.frozen).resolve()
        if fp != FROZEN.resolve() and not (a.campaign and a.campaign.startswith("DRYRUN_")
                                           and fp.parent.name.startswith("DRYRUN_")
                                           and fp.parent.parent == (E4 / "runs").resolve()):
            print(f"REFUSING: --frozen {fp} (a dry-run record must sit in runs/DRYRUN_*/ and serve a DRYRUN_ campaign)")
            sys.exit(2)
    res = check(Path(a.registration), a.stage, a.campaign, Path(a.frozen) if a.frozen else None)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "tables"}, indent=1))
    print(json.dumps(res["tables"]))
    if res.get("malformed"):
        sys.exit(2)
    sys.exit(1 if res["differences"] else 0)


if __name__ == "__main__":
    main()
