#!/usr/bin/env python
"""R19, gate G4: check the finished test runs against the freeze, before the reports are read.

    .venv-dl/bin/python scripts/r19_check.py | tee logs/r19_check.out

Read only. One line per check; stops at the first failure. Checks:
- the registration text and every frozen source under src/ and scripts/ are unchanged since the freeze
  (this file is the one new source);
- the seed files are exactly the 22 jobs of logs/jobs_R19.txt (registration, name, arm, parts, encoder,
  clauses, seed), and every file of the six arms was written after the freeze (json time and file mtime);
- TM cfg equals the bge-large reference arm's (R18 tm_mem, R9b tm_mem_short) but for the seed;
- bits: train and dev hashes equal the e60/e61 dev rows (results/explore/abcd.jsonl); the test hash is the
  same over seeds and between tm_mem_static and lr_mem_static; bit_names.json equals the dry run's;
- seed 1 (TM) and seed 0 (LR) give the dev accuracy of their e60/e61 rows exactly;
- each seed npz gives its json's test accuracy;
- the copied references: every file's SHA-256 equals logs/r19_refs.out and its source in R18 or R9b, and
  no other file is in those folders;
- the test rows: y and convo_id equal in every arm; 611 STAR and 1,004 ABCD conversations.
Added after the R19 runs (2026-10-18).
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
FROZEN = ROOT / "registrations" / "R19_no_network_agent.frozen.json"
REF_CFG = {"R19star": CONF / "R18" / "tm_mem" / "seed1.json", "R19abcd": CONF / "R9b" / "tm_mem_short" / "seed1.json"}
DEV_ROW = {("R19star", "tm_mem_static"): ("e60_star_B_c16000", "tm"), ("R19star", "tm_nomem_static"): ("e61_star_Bnomem_c16000", "tm"),
           ("R19star", "lr_mem_static"): ("e60_star_B_c4000", "lr_bits"), ("R19abcd", "tm_mem_static"): ("e60_abcd_B_c4000", "tm"),
           ("R19abcd", "tm_nomem_static"): ("e61_abcd_Bnomem_c4000", "tm"), ("R19abcd", "lr_mem_static"): ("e60_abcd_B_c4000", "lr_bits")}
N_CONVOS = {"R19star": 611, "R19abcd": 1004}
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
    changed = sorted(f for f in set(now) | set(rec["sources"]) if rec["sources"].get(f) != now.get(f))
    need(changed == ["scripts/r19_check.py"], f"all {len(rec['sources'])} frozen sources unchanged (new since: {changed})")

    jobs = {}
    for line in (ROOT / "logs" / "jobs_R19.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arg = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]  # noqa: E731
        key = (arg("registration"), arg("name"), int(arg("seeds")))
        jobs[key] = {"dataset": arg("dataset"), "encoder": arg("encoder"), "arm": arg("arm"), "parts": arg("parts").split(","),
                     "clauses": int(arg("clauses")) if arg("clauses") else None}
    found = {(r, p.parent.name, int(p.stem[4:])) for r in ("R19star", "R19abcd") for p in (CONF / r).glob("*_static/seed*.json")}
    need(len(jobs) == 22 and found == set(jobs), f"seed files are exactly the {len(jobs)} jobs of logs/jobs_R19.txt")

    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if "tag" in d:
            rows[(d["tag"], d.get("arm"))] = d
    meta0 = {}
    for (reg, name, seed), job in sorted(jobs.items()):
        folder = CONF / reg / name
        j = json.loads((folder / f"seed{seed}.json").read_text())
        tag = f"{reg}/{name}/seed{seed}"
        need(j["registration"] == reg and j["name"] == name and j["seed"] == seed and j["arm"] == job["arm"]
             and j["parts"] == job["parts"] and j["encoder"] == job["encoder"] == "potion-32m", f"{tag}: settings as the job line")
        t_json = calendar.timegm(time.strptime(j["time"], "%Y-%m-%dT%H:%M:%SZ"))
        mtimes = [p.stat().st_mtime for p in folder.iterdir()]
        need(t_json > t_freeze and min(mtimes) > t_freeze, f"{tag}: written {j['time']}, every file of the arm after the freeze {rec['frozen_utc']}")
        if job["arm"] == "tm":
            ref = json.loads(REF_CFG[reg].read_text())["cfg"]
            need({**j["cfg"], "seed": None} == {**ref, "seed": None} and j["cfg"]["seed"] == seed and j["cfg"]["n_clauses"] == job["clauses"],
                 f"{tag}: cfg equals {REF_CFG[reg].parent.parent.name} {REF_CFG[reg].parent.name}'s but the seed ({j['cfg']['n_clauses']} clauses, best epoch {j['best_epoch']})")
        else:
            ok(f"{tag}: LR C {j['C']} (grid 0.1 to 100)")
        row = rows[DEV_ROW[(reg, name)]]
        need(j["bits_sha256"]["train"] == row["bits_sha256"]["train"] and j["bits_sha256"]["dev"] == row["bits_sha256"]["dev"],
             f"{tag}: train/dev bits equal {DEV_ROW[(reg, name)][0]}'s ({row['bits_sha256']['train'][:8]}/{row['bits_sha256']['dev'][:8]})")
        if seed in (0, 1):
            need(j["dev_acc"] == row["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals the dev row's exactly")
        z = np.load(folder / f"seed{seed}.npz")
        need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy {j['test_acc']:.6f}")
        m = meta0.setdefault(reg, {"y": z["y"], "convo_id": z["convo_id"], "test_bits": {}})
        need(np.array_equal(z["y"], m["y"]) and np.array_equal(z["convo_id"], m["convo_id"]), f"{tag}: test rows equal the other arms'")
        m["test_bits"].setdefault(tuple(j["parts"]), set()).add(j["bits_sha256"]["test"])
    for reg, m in meta0.items():
        need(all(len(v) == 1 for v in m["test_bits"].values()), f"{reg}: one test-bit hash per view over seeds and arms "
             + ", ".join(f"{'+'.join(k)} {next(iter(v))[:8]}" for k, v in m["test_bits"].items()))
        need(len(np.unique(m["convo_id"])) == N_CONVOS[reg], f"{reg}: {len(m['y'])} test action turns in {N_CONVOS[reg]} conversations")
        for name in ("tm_mem_static", "tm_nomem_static", "lr_mem_static"):
            dry = ROOT / "results" / "explore" / "dryrun" / f"dry_{reg}_{name}" / "bit_names.json"
            need(sha(CONF / reg / name / "bit_names.json") == sha(dry), f"{reg}/{name}: bit_names.json equals the dry run's")
        need(sha(CONF / reg / "tm_mem_static" / "bit_names.json") == sha(CONF / reg / "lr_mem_static" / "bit_names.json"), f"{reg}: TM and LR read the same bits")

    listed = {}
    for line in (ROOT / "logs" / "r19_refs.out").read_text().splitlines():
        path, h, src = line.split(" ", 2)
        listed[path] = (h, src.strip("()").removeprefix("from "))
    for path, (h, src) in listed.items():
        dst = CONF / path
        need(sha(dst) == h and sha(CONF / src / dst.name) == h, f"copy {path} equals the list and its source {src}/{dst.name}")
    present = {str(p.relative_to(CONF)) for r in ("R19star", "R19abcd") for p in (CONF / r).glob("*_R*/*") if not p.parent.name.endswith("_static")}
    need(present == set(listed) and len(listed) == 70, f"the reference folders hold exactly the {len(listed)} listed files")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
