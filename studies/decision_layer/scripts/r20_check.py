#!/usr/bin/env python
"""R20: check the finished test runs against the freeze, before the reports are read.

    .venv-dl/bin/python scripts/r20_check.py | tee logs/r20_check.out

Read only. One line per check; stops at the first failure. Written before R20's freeze (2026-10-18), so it is one
of the frozen sources. Checks:
- the registration text and every frozen source under src/ and scripts/ are unchanged since the freeze (new files
  are listed: no frozen file imports them, so they cannot be on R20's path);
- the seed files are exactly the jobs of logs/jobs_R20.txt: one per (registration, name, seed) of every
  abcd_confirm.py command and one per (out, name + "_closed", seed) of every abcd_closed_loop.py command; every
  file of these arms was written after the freeze (json time and file mtime);
- TM cfg equals R19's tm_mem_static of the same dataset but for the seed; the clause count is the job line's;
- LR: the json's C grid equals the job's --grid-c, and C and the dev accuracy equal the dry run's;
- bits: train and dev hashes equal the e65 dev rows (results/explore/abcd.jsonl); one test hash per view over
  seeds and arms; bit_names.json equals the dry run's; seed 1 (TM) gives its e65 row's dev accuracy exactly;
- each seed npz gives its json's test accuracy;
- each closed-loop seed: its open-loop accuracy equals the test accuracy of the same model's open-loop arm
  (exactly for the TM; within 2 test rows for LR, whose closed loop scores in numpy on the CPU), and its npz
  gives its json's closed-loop accuracy;
- the copied references: every file's SHA-256 equals logs/r20_refs.out and its source, and no other file is in
  those folders;
- the test rows: y and convo_id equal in every arm; 611 STAR and 1,004 ABCD conversations.
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
DRY = ROOT / "results" / "explore" / "dryrun"
FROZEN = ROOT / "registrations" / "R20_customer_only_closed_loop.frozen.json"
REGS = ("R20star", "R20abcd")
REF_CFG = {"R20star": CONF / "R19star" / "tm_mem_static" / "seed1.json", "R20abcd": CONF / "R19abcd" / "tm_mem_static" / "seed1.json"}
DEV_ROW = {("R20star", "tm_mem_cust"): ("e65_star_C_c16000", "tm"), ("R20star", "tm_nomem_cust"): ("e65_star_Cnomem_c16000", "tm"),
           ("R20star", "lr_mem_cust"): ("e65_star_C_c16000", "lr_bits"), ("R20abcd", "tm_mem_cust"): ("e65_abcd_C_c4000", "tm"),
           ("R20abcd", "tm_nomem_cust"): ("e65_abcd_Cnomem_c4000", "tm"), ("R20abcd", "lr_mem_cust"): ("e65_abcd_C_c4000", "lr_bits")}
N_CONVOS = {"R20star": 611, "R20abcd": 1004}
REF_SUFFIX = ("_R19", "_R18", "_R13")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def ok(msg: str) -> None:
    print("ok  " + msg, flush=True)


def need(cond: bool, msg: str) -> None:
    if not cond:
        print("FAIL " + msg, flush=True)
        sys.exit(1)
    ok(msg)


def parse_jobs() -> tuple[dict, dict]:
    """abcd_confirm.py commands keyed (registration, name, seed); abcd_closed_loop.py commands keyed (out, name_closed, seed)."""
    confirm, closed = {}, {}
    for line in (ROOT / "logs" / "jobs_R20.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for cmd in line.split(" && "):
            arg = lambda k: (re.search(rf"--{k} (\S+)", cmd) or [None, None])[1]  # noqa: E731
            seeds = [int(s) for s in re.search(r"--seeds ((?:\d+ ?)+)", cmd)[1].split()]
            if "scripts/abcd_confirm.py" in cmd:
                for s in seeds:
                    confirm[(arg("registration"), arg("name"), s)] = {
                        "dataset": arg("dataset"), "encoder": arg("encoder"), "arm": arg("arm"), "parts": arg("parts").split(","),
                        "clauses": int(arg("clauses")) if arg("clauses") else None,
                        "grid_c": [float(c) for c in arg("grid-c").split(",")] if arg("grid-c") else None}
            elif "scripts/abcd_closed_loop.py" in cmd:
                assert arg("split") == "test" and arg("out") in REGS, cmd
                for s in seeds:
                    closed[(arg("out"), arg("name") + "_closed", s)] = {"src": arg("registration"), "name": arg("name"), "kind": arg("kind"),
                                                                         "dataset": arg("dataset"), "parts": arg("parts").split(",")}
            else:
                raise SystemExit(f"unknown command in jobs_R20.txt: {cmd}")
    return confirm, closed


def main() -> None:
    rec = json.loads(FROZEN.read_text())
    t_freeze = calendar.timegm(time.strptime(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    need(sha(ROOT / rec["registration"]) == rec["sha256"], f"registration text unchanged ({rec['sha256'][:16]})")
    now = {str(p.relative_to(ROOT)): sha(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
    changed = sorted(f for f in rec["sources"] if rec["sources"][f] != now.get(f))
    new = sorted(set(now) - set(rec["sources"]))
    need(changed == [], f"all {len(rec['sources'])} frozen sources unchanged (changed or gone: {changed}; new since, not on R20's path: {new})")

    confirm, closed = parse_jobs()
    want = set(confirm) | set(closed)
    found = {(r, p.parent.name, int(p.stem[4:])) for r in REGS for p in (CONF / r).glob("*/seed*.json") if not p.parent.name.endswith(REF_SUFFIX)}
    need(found == want, f"seed files are exactly the {len(confirm)} training and {len(closed)} closed-loop seeds of logs/jobs_R20.txt")
    for reg, name in sorted({(r, n) for r, n, _ in want}):
        folder = CONF / reg / name
        times = [calendar.timegm(time.strptime(json.loads(p.read_text())["time"], "%Y-%m-%dT%H:%M:%SZ")) for p in folder.glob("seed*.json")]
        need(min(times) > t_freeze and min(p.stat().st_mtime for p in folder.iterdir()) > t_freeze,
             f"{reg}/{name}: every json time and file of the arm after the freeze {rec['frozen_utc']}")

    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if "tag" in d and "script" not in d:
            rows[(d["tag"], d.get("arm"))] = d
    meta0, test_bits = {}, {}

    def same_rows(reg: str, z, tag: str) -> None:
        m = meta0.setdefault(reg, {"y": z["y"], "convo_id": z["convo_id"]})
        need(np.array_equal(z["y"], m["y"]) and np.array_equal(z["convo_id"], m["convo_id"]), f"{tag}: test rows equal the other arms'")

    open_acc = {}
    for (reg, name, seed), job in sorted(confirm.items()):
        folder = CONF / reg / name
        j = json.loads((folder / f"seed{seed}.json").read_text())
        tag = f"{reg}/{name}/seed{seed}"
        need(j["registration"] == reg and j["name"] == name and j["seed"] == seed and j["arm"] == job["arm"]
             and j["parts"] == job["parts"] and j["encoder"] == job["encoder"] == "potion-32m", f"{tag}: settings as the job line")
        dry = json.loads((DRY / f"dry_{reg}_{name}" / f"seed{seed}.json").read_text()) if (DRY / f"dry_{reg}_{name}" / f"seed{seed}.json").exists() else None
        if job["arm"] == "tm":
            ref = json.loads(REF_CFG[reg].read_text())["cfg"]
            need({**j["cfg"], "seed": None} == {**ref, "seed": None} and j["cfg"]["seed"] == seed and j["cfg"]["n_clauses"] == job["clauses"],
                 f"{tag}: cfg equals R19's tm_mem_static but the seed ({j['cfg']['n_clauses']} clauses, best epoch {j['best_epoch']})")
        else:
            need(job["grid_c"] is not None and j.get("grid_c") == job["grid_c"] and j["C"] in job["grid_c"],
                 f"{tag}: LR C {j['C']} from the grid {job['grid_c']}")
            need(dry is not None and j["C"] == dry["C"] and j["dev_acc"] == dry["dev_acc"],
                 f"{tag}: C and dev accuracy {j['dev_acc']:.6f} equal the dry run's")
        row = rows[DEV_ROW[(reg, name)]]
        need(j["bits_sha256"]["train"] == row["bits_sha256"]["train"] and j["bits_sha256"]["dev"] == row["bits_sha256"]["dev"],
             f"{tag}: train/dev bits equal {DEV_ROW[(reg, name)][0]}'s ({row['bits_sha256']['train'][:8]}/{row['bits_sha256']['dev'][:8]})")
        if job["arm"] == "tm" and seed == 1:
            need(j["dev_acc"] == row["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals the e65 row's exactly")
        z = np.load(folder / f"seed{seed}.npz")
        need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy {j['test_acc']:.6f}")
        same_rows(reg, z, tag)
        test_bits.setdefault((reg, tuple(j["parts"])), set()).add(j["bits_sha256"]["test"])
        open_acc[(reg, name, seed)] = j["test_acc"]
    for (reg, parts), v in sorted(test_bits.items()):
        need(len(v) == 1, f"{reg} {'+'.join(parts)}: one test-bit hash over seeds and arms ({next(iter(v))[:8]})")
    for reg, name in sorted({(r, n) for r, n, _ in confirm}):
        need(sha(CONF / reg / name / "bit_names.json") == sha(DRY / f"dry_{reg}_{name}" / "bit_names.json"), f"{reg}/{name}: bit_names.json equals the dry run's")

    for (out, cname, seed), job in sorted(closed.items()):
        folder = CONF / out / cname
        j = json.loads((folder / f"seed{seed}.json").read_text())
        tag = f"{out}/{cname}/seed{seed}"
        need(j["registration"] == job["src"] and j["name"] == job["name"] and j["kind"] == job["kind"] and j["split"] == "test" and j["seed"] == seed,
             f"{tag}: settings as the job line (models of {job['src']}/{job['name']})")
        src_json = json.loads((CONF / job["src"] / job["name"] / f"seed{seed}.json").read_text())
        n = len(np.load(folder / f"seed{seed}.npz")["y"])
        gap = abs(j["acc_open"] - src_json["test_acc"]) * n
        need(gap < 1e-6 if job["kind"] == "tm" else gap < 2.5,
             f"{tag}: open-loop accuracy {j['acc_open']:.6f} equals {job['src']}/{job['name']}'s test accuracy ({gap:.0f} rows apart)")
        z = np.load(folder / f"seed{seed}.npz")
        need(abs(float((z["pred"] == z["y"]).mean()) - j["acc_closed"]) < 1e-12, f"{tag}: npz gives the json's closed-loop accuracy {j['acc_closed']:.6f}")
        same_rows(out, z, tag)

    listed = {}
    for line in (ROOT / "logs" / "r20_refs.out").read_text().splitlines():
        path, h, src = line.split(" ", 2)
        listed[path] = (h, src.strip("()").removeprefix("from "))
    for path, (h, src) in sorted(listed.items()):
        dst = CONF / path
        need(sha(dst) == h and sha(CONF / src / dst.name) == h, f"copy {path} equals the list and its source {src}/{dst.name}")
        if dst.suffix == ".npz":
            same_rows(path.split("/")[0], np.load(dst), path)
    present = {str(p.relative_to(CONF)) for r in REGS for p in (CONF / r).glob("*/*") if p.parent.name.endswith(REF_SUFFIX)}
    need(present == set(listed), f"the reference folders hold exactly the {len(listed)} listed files")
    for reg, m in sorted(meta0.items()):
        need(len(np.unique(m["convo_id"])) == N_CONVOS[reg], f"{reg}: {len(m['y'])} test action turns in {N_CONVOS[reg]} conversations")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
