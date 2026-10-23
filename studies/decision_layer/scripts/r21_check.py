#!/usr/bin/env python
"""R21 gate: check the finished test runs against the freeze, before the reports are read.

    .venv-dl/bin/python scripts/r21_check.py | tee logs/r21_check.out

Read only. One line per check; stops at the first failure. Written while the R21 runs were going, before any test
number was read. Checks:
- the registration text is unchanged since the freeze;
- every source on R21's path is unchanged (count_text, table_run, seq_queue, abcd_confirm, report_abcd,
  static_text, abcd, tm, tm_init, boolean, linear). Other changes since the freeze are listed and must be in the
  allowed set: this checker, the e72d / e73 scripts, and e71's dated fix of verify_enum.py, all off R21's path;
- the seed files are exactly the 22 jobs of logs/jobs_R21.txt (registration, name, arm, parts, clauses, seed, grid),
  with encoder none and the C4 table (file e1771d9e) in static_info, and every file of the three arms written after
  the freeze;
- TM cfg equals R19's tm_mem_static of the same dataset but for the seed; LR's grid is the wide one;
- bits: train and dev hashes equal the dry run of the same view, and the e72 dev rows where they exist; the test
  hash is the same over seeds and between tm_mem_c4 and lr_mem_c4; bit_names.json equals the dry run's;
- seed 1 (TM) gives the dev accuracy of its e72 row exactly where one exists; LR seed 0 gives its dry run's dev accuracy;
- each seed npz gives its json's test accuracy; y and convo_id equal in every arm, references included (611 STAR and
  1,004 ABCD conversations);
- the copied references: every file's SHA-256 equals logs/r21_refs.out and its source, and the reference folders hold
  exactly the 120 listed files.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
FROZEN = ROOT / "registrations" / "R21_c4_table_agent.frozen.json"
TABLE_FILE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
WIDE = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
ON_PATH = ["src/dl/count_text.py", "scripts/table_run.py", "scripts/seq_queue.sh", "scripts/abcd_confirm.py", "scripts/report_abcd.py",
           "src/dl/static_text.py", "src/dl/abcd.py", "src/dl/tm.py", "src/dl/tm_init.py", "src/dl/boolean.py", "src/dl/linear.py"]
ALLOWED = {"scripts/r21_check.py", "scripts/verify_policy_c4.py", "scripts/e71_offwindow_precheck.py", "scripts/star_replies.py",
           "scripts/verify_enum.py", "scripts/e71_k35_sample.py"}  # e71_k35_sample.py added 11:15 UTC (written after this checker; e71, off the path)
E72_ROW = {("R21star", "tm_mem_c4"): "e72_star_Bc4_c16000", ("R21abcd", "tm_mem_c4"): "e72_abcd_Bc4_c4000",
           ("R21abcd", "tm_nomem_c4"): "e72_abcd_Bc4nomem_c4000"}
N_CONVOS = {"R21star": 611, "R21abcd": 1004}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def ok(msg: str) -> None:
    print("ok  " + msg, flush=True)


def need(cond: bool, msg: str) -> None:
    if not cond:
        print("FAIL " + msg, flush=True)
        sys.exit(1)
    ok(msg)


def main() -> None:
    rec = json.loads(FROZEN.read_text())
    t_freeze = calendar.timegm(time.strptime(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    need(sha(ROOT / rec["registration"]) == rec["sha256"], f"registration text unchanged ({rec['sha256'][:16]})")
    now = {str(p.relative_to(ROOT)): sha(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
    for f in ON_PATH:
        need(now.get(f) == rec["sources"].get(f) is not None, f"on R21's path, unchanged: {f} ({now[f][:8]})")
    changed = sorted(f for f in set(now) | set(rec["sources"]) if rec["sources"].get(f) != now.get(f))
    need(set(changed) <= ALLOWED, f"other changes since the freeze, all off R21's path: {changed}")

    jobs = {}
    for line in (ROOT / "logs" / "jobs_R21.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arg = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]  # noqa: E731
        key = (arg("registration"), arg("name"), int(arg("seeds")))
        jobs[key] = {"dataset": arg("dataset"), "encoder": arg("encoder"), "arm": arg("arm"), "parts": arg("parts").split(","),
                     "clauses": int(arg("clauses")) if arg("clauses") else None, "grid": arg("grid-c")}
    found = {(r, p.parent.name, int(p.stem[4:])) for r in ("R21star", "R21abcd") for p in (CONF / r).glob("*_c4/seed*.json")}
    need(len(jobs) == 22 and found == set(jobs), f"seed files are exactly the {len(jobs)} jobs of logs/jobs_R21.txt")

    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and "tag" in d:
            rows[d["tag"]] = d
    meta0 = {}
    for (reg, name, seed), job in sorted(jobs.items()):
        folder = CONF / reg / name
        j = json.loads((folder / f"seed{seed}.json").read_text())
        tag = f"{reg}/{name}/seed{seed}"
        need(j["registration"] == reg and j["name"] == name and j["seed"] == seed and j["arm"] == job["arm"] and j["parts"] == job["parts"]
             and j["encoder"] == job["encoder"] == "none", f"{tag}: settings as the job line (encoder none)")
        need(all(v["table_file_sha256"] == TABLE_FILE and v["table"] == "c4words-S1000-v1" for v in j["static_info"].values()),
             f"{tag}: the C4 table in {sorted(j['static_info'])}")
        t_json = calendar.timegm(time.strptime(j["time"], "%Y-%m-%dT%H:%M:%SZ"))
        mtimes = [p.stat().st_mtime for p in folder.iterdir()]
        need(t_json > t_freeze and min(mtimes) > t_freeze, f"{tag}: written {j['time']}, every file of the arm after the freeze {rec['frozen_utc']}")
        if job["arm"] == "tm":
            ref = json.loads((CONF / reg.replace("R21", "R19") / "tm_mem_static" / "seed1.json").read_text())["cfg"]
            need({**j["cfg"], "seed": None} == {**ref, "seed": None} and j["cfg"]["seed"] == seed and j["cfg"]["n_clauses"] == job["clauses"],
                 f"{tag}: cfg equals R19's tm_mem_static but the seed ({j['cfg']['n_clauses']} clauses, best epoch {j['best_epoch']})")
        else:
            need(j.get("grid_c") == WIDE, f"{tag}: LR on the wide grid, C {j['C']}")
        dry = json.loads((ROOT / "results" / "explore" / "dryrun" / f"dry_{reg}_{name}" / f"seed{0 if job['arm'] == 'lr' else 1}.json").read_text())
        need(j["bits_sha256"]["train"] == dry["bits_sha256"]["train"] and j["bits_sha256"]["dev"] == dry["bits_sha256"]["dev"],
             f"{tag}: train/dev bits equal the dry run's ({dry['bits_sha256']['train'][:8]}/{dry['bits_sha256']['dev'][:8]})")
        e72 = rows.get(E72_ROW.get((reg, name), ""))
        if e72 is not None:
            need(j["bits_sha256"]["train"] == e72["bits_sha256"]["train"] and j["bits_sha256"]["dev"] == e72["bits_sha256"]["dev"],
                 f"{tag}: train/dev bits equal {E72_ROW[(reg, name)]}'s")
            if seed == 1:
                need(j["dev_acc"] == e72["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals {E72_ROW[(reg, name)]}'s exactly")
        if job["arm"] == "lr":
            need(j["dev_acc"] == dry["dev_acc"] and j["C"] == dry["C"], f"{tag}: dev accuracy {j['dev_acc']:.6f} and C {j['C']} equal the dry run's")
        z = np.load(folder / f"seed{seed}.npz")
        need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy")
        m = meta0.setdefault(reg, {"y": z["y"], "convo_id": z["convo_id"], "test_bits": {}})
        need(np.array_equal(z["y"], m["y"]) and np.array_equal(z["convo_id"], m["convo_id"]), f"{tag}: test rows equal the other arms'")
        m["test_bits"].setdefault(tuple(j["parts"]), set()).add(j["bits_sha256"]["test"])
    for reg, m in meta0.items():
        need(all(len(v) == 1 for v in m["test_bits"].values()), f"{reg}: one test-bit hash per view over seeds and arms "
             + ", ".join(f"{'+'.join(k)} {next(iter(v))[:8]}" for k, v in m["test_bits"].items()))
        need(len(np.unique(m["convo_id"])) == N_CONVOS[reg], f"{reg}: {len(m['y'])} test action turns in {N_CONVOS[reg]} conversations")
        for name in ("tm_mem_c4", "tm_nomem_c4", "lr_mem_c4"):
            dryn = ROOT / "results" / "explore" / "dryrun" / f"dry_{reg}_{name}" / "bit_names.json"
            need(sha(CONF / reg / name / "bit_names.json") == sha(dryn), f"{reg}/{name}: bit_names.json equals the dry run's")
        need(sha(CONF / reg / "tm_mem_c4" / "bit_names.json") == sha(CONF / reg / "lr_mem_c4" / "bit_names.json"), f"{reg}: TM and LR read the same bits")
        for p in sorted((CONF / reg).glob("*_R*/seed*.npz")):
            z = np.load(p)
            need(np.array_equal(z["y"], m["y"]) and np.array_equal(z["convo_id"], m["convo_id"]), f"{reg}: reference {p.parent.name}/{p.name} has the same test rows")

    listed = {}
    for line in (ROOT / "logs" / "r21_refs.out").read_text().splitlines():
        path, h, src = line.split(" ", 2)
        listed[path] = (h, src.strip("()").removeprefix("from "))
    for path, (h, src) in listed.items():
        dst = CONF / path
        need(sha(dst) == h and sha(CONF / src / dst.name) == h, f"copy {path} equals the list and its source {src}/{dst.name}")
    present = {str(p.relative_to(CONF)) for r in ("R21star", "R21abcd") for p in (CONF / r).glob("*/*") if not p.parent.name.endswith("_c4")}
    need(present == set(listed) and len(listed) == 120, f"the reference folders hold exactly the {len(listed)} listed files")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
