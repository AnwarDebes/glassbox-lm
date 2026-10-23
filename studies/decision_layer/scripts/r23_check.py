#!/usr/bin/env python
"""R23 gate: check the finished test runs against the freeze, before the report is read.

    .venv-dl/bin/python scripts/r23_check.py | tee logs/r23_check.out

Read only. One line per check; stops at the first failure. Written before any R23 test number was read (adapted from
scripts/r22_check.py). Checks:
- the registration text is unchanged since the freeze; every source on R23's path is unchanged, and other changes since
  the freeze are only this checker;
- the seed files are exactly the 10 jobs of logs/jobs_R23.txt, written after the freeze. The C4 arm has encoder none
  and the C4 table (file e1771d9e) in static_info; the bge-large arm has encoder bge-large and R9b's parts;
- cfg: the C4 arm equals R21abcd's tm_mem_c4 and the bge-large arm equals R9b's tm_mem_short, but for s (16 / 24) and the seed;
- bits: train and dev hashes equal the dry runs' and the dev rows'; the test hash is one per arm over seeds, and the
  C4 arm's equals R21abcd's;
- seed 1 gives the dev accuracy of its dev row exactly (e74c_abcd_Bc4_c4000_s16, e75_abcd_bge_c4000_s24);
- each npz gives its json's test accuracy; y and convo_id equal in every arm and reference (1,004 dialogues);
- the copied references: every file's SHA-256 equals logs/r23_refs.out and its source; the folders hold exactly the
  32 listed files.
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
REG = "R23abcd"
FROZEN = ROOT / "registrations" / "R23_tuned_s_abcd.frozen.json"
TABLE_FILE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
ON_PATH = ["src/dl/count_text.py", "scripts/table_run.py", "scripts/seq_queue.sh", "scripts/abcd_confirm.py", "scripts/report_abcd.py",
           "src/dl/static_text.py", "src/dl/abcd.py", "src/dl/tm.py", "src/dl/tm_init.py", "src/dl/boolean.py", "src/dl/linear.py"]
ALLOWED = {"scripts/r23_check.py"}
ARMS = {"tm_mem_c4_s16": {"encoder": "none", "parts": ["mem", "words", "xcust", "xrecent"], "cfg_ref": CONF / "R21abcd" / "tm_mem_c4" / "seed1.json",
                          "e74": "e74c_abcd_Bc4_c4000_s16", "dry": "dry_R23abcd_tm_mem_c4_s16", "test_ref": CONF / "R21abcd" / "tm_mem_c4" / "seed1.json", "s": 16.0},
        "tm_mem_bge_s24": {"encoder": "bge-large", "parts": ["mem", "intent", "step", "words"], "cfg_ref": CONF / "R9b" / "tm_mem_short" / "seed1.json",
                           "e74": "e75_abcd_bge_c4000_s24", "dry": "dry_R23abcd_tm_mem_bge_s24", "test_ref": None, "s": 24.0}}
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
        need(now.get(f) == rec["sources"].get(f) is not None, f"on R23's path, unchanged: {f} ({now[f][:8]})")
    changed = sorted(f for f in set(now) | set(rec["sources"]) if rec["sources"].get(f) != now.get(f))
    need(set(changed) <= ALLOWED, f"other changes since the freeze: {changed}")

    jobs = {}
    for line in (ROOT / "logs" / "jobs_R23.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arg = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]  # noqa: E731
        jobs[(arg("name"), int(arg("seeds")))] = {"encoder": arg("encoder"), "parts": arg("parts").split(","), "clauses": int(arg("clauses")),
                                                  "s": float(arg("s"))}
    found = {(p.parent.name, int(p.stem[4:])) for name in ARMS for p in (CONF / REG / name).glob("seed*.json")}
    need(len(jobs) == 10 and found == set(jobs), f"seed files are exactly the {len(jobs)} jobs of logs/jobs_R23.txt")

    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and "tag" in d:
            rows[d["tag"]] = d
    meta0, test_bits = None, {}
    for (name, seed), job in sorted(jobs.items()):
        a = ARMS[name]
        folder = CONF / REG / name
        j = json.loads((folder / f"seed{seed}.json").read_text())
        tag = f"{REG}/{name}/seed{seed}"
        need(j["registration"] == REG and j["name"] == name and j["seed"] == seed and j["arm"] == "tm" and j["parts"] == job["parts"] == a["parts"]
             and j["encoder"] == job["encoder"] == a["encoder"], f"{tag}: settings as the job line (encoder {a['encoder']})")
        if a["encoder"] == "none":
            need(all(v["table_file_sha256"] == TABLE_FILE for v in j["static_info"].values()), f"{tag}: the C4 table in {sorted(j['static_info'])}")
        t_json = calendar.timegm(time.strptime(j["time"], "%Y-%m-%dT%H:%M:%SZ"))
        mtimes = [p.stat().st_mtime for p in folder.iterdir()]
        need(t_json > t_freeze and min(mtimes) > t_freeze, f"{tag}: written {j['time']}, every file of the arm after the freeze")
        ref = json.loads(a["cfg_ref"].read_text())["cfg"]
        need({**j["cfg"], "seed": None, "s": None} == {**ref, "seed": None, "s": None} and j["cfg"]["s"] == a["s"] == job["s"] and j["cfg"]["seed"] == seed
             and j["cfg"]["n_clauses"] == 4000, f"{tag}: cfg equals {a['cfg_ref'].parent.parent.name}/{a['cfg_ref'].parent.name}'s but s {a['s']:g} and the seed (best epoch {j['best_epoch']})")
        dry = json.loads((ROOT / "results" / "explore" / "dryrun" / a["dry"] / "seed1.json").read_text())
        e74 = rows[a["e74"]]
        need(j["bits_sha256"]["train"] == dry["bits_sha256"]["train"] == e74["bits_sha256"]["train"]
             and j["bits_sha256"]["dev"] == dry["bits_sha256"]["dev"] == e74["bits_sha256"]["dev"],
             f"{tag}: train/dev bits equal the dry run's and {a['e74']}'s ({e74['bits_sha256']['train'][:8]}/{e74['bits_sha256']['dev'][:8]})")
        if seed == 1:
            need(j["dev_acc"] == e74["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals {a['e74']}'s exactly")
        z = np.load(folder / f"seed{seed}.npz")
        need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy")
        if meta0 is None:
            meta0 = {"y": z["y"], "convo_id": z["convo_id"]}
        need(np.array_equal(z["y"], meta0["y"]) and np.array_equal(z["convo_id"], meta0["convo_id"]), f"{tag}: test rows equal the other arms'")
        test_bits.setdefault(name, set()).add(j["bits_sha256"]["test"])
    for name, hs in test_bits.items():
        need(len(hs) == 1, f"{name}: one test-bit hash over seeds ({next(iter(hs))[:8]})")
        if ARMS[name]["test_ref"] is not None:
            r21 = json.loads(ARMS[name]["test_ref"].read_text())["bits_sha256"]["test"]
            need(next(iter(hs)) == r21, f"{name}: test bits equal R21abcd's tm_mem_c4 ({r21[:8]})")
    need(len(np.unique(meta0["convo_id"])) == 1004, f"{len(meta0['y'])} test action turns in 1004 conversations")
    for p in sorted((CONF / REG).glob("*_R*/seed*.npz")):
        z = np.load(p)
        need(np.array_equal(z["y"], meta0["y"]) and np.array_equal(z["convo_id"], meta0["convo_id"]), f"reference {p.parent.name}/{p.name} has the same test rows")
    listed = {}
    for line in (ROOT / "logs" / "r23_refs.out").read_text().splitlines():
        path, h, src = line.split(" ", 2)
        listed[path] = (h, src.strip("()").removeprefix("from "))
    for path, (h, src) in listed.items():
        dst = CONF / path
        need(sha(dst) == h and sha(CONF / src / dst.name) == h, f"copy {path} equals the list and its source {src}/{dst.name}")
    present = {str(p.relative_to(CONF)) for p in (CONF / REG).glob("*_R*/*")}
    need(present == set(listed) and len(listed) == 32, f"the reference folders hold exactly the {len(listed)} listed files")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
