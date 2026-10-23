#!/usr/bin/env python
"""R24 gate: check the finished MultiWOZ test runs against the freeze, before the report is read (adapted from
scripts/r23_check.py; written before any R24 test number).

    .venv-dl/bin/python scripts/r24_check.py | tee logs/r24_check.out        # after logs/jobs_R24.txt
    .venv-dl/bin/python scripts/r24_check.py --dry | tee logs/R24_dry_check.out   # after logs/jobs_R24_dry.txt, before the freeze
    .venv-dl/bin/python scripts/r24_check.py --gate > logs/R24_gate.out          # step 0 of logs/R24_chain.sh, before any test file

Read only. One line per check; stops at the first failure.
- the registration is unchanged since the freeze, and so is every source on R24's path; no other file under src/ or
  scripts/ was added, changed or removed since the freeze (this checker included);
- each TM arm runs at its agent's s* from e85 (logs/e85_pick_final.out); logs/jobs_R24.txt equals e85_pick.py's
  r24_jobs(s*) line for line;
- the seed files are exactly the 18 jobs of logs/jobs_R24.txt, written after the freeze;
- each TM arm's settings, init (half) and cfg equal its 16k seed-1 dev row's at s* (e83b / e84 at s 8, else e85b; random
  negatives), apart from the seed, and its table is the arm's;
- train and dev bits equal the dry run's and the dev row's; one test-bit hash per arm over seeds; the LR arms' bits
  equal their TM arm's;
- seed 1's dev accuracy equals its dev row's exactly; each LR arm's dev accuracy and C equal its dev LR row's;
- each npz gives its json's test accuracy, and y, convo_id, k, w_bert and w_roberta are equal in every arm; the arms hold
  every test turn and dialogue of examples_test.json.gz (MultiWOZ 2.2 test has 1,000 dialogues);
- examples_test.json.gz was written after the freeze with train's label list; the bge-large dev-stage file has its
  frozen hash, and the new file's train and dev arrays equal it (the provenance entry aside, which --add-test rewrites);
- it prints how many e85 rows were written before the freeze (e85 started before it; a note, not a check).
--dry checks the six dry runs instead: settings, train and dev bits equal to the dev rows', test bits equal to dev
bits (dev stands in for test), and the LR dry runs' dev accuracy and C equal the dev LR rows'.
--gate prints G-s* (each network-free agent's 16k seed-1 dev row at s* minus bge-large's, at least -1.0) and exits 3
if both agents miss it. Tested by tests/test_r24_check.py (made-up files).
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e85_pick import r24_jobs  # noqa: E402

CONF, DRY = ROOT / "results" / "confirm", ROOT / "results" / "explore" / "dryrun"
REG = "R24mwoz"
FROZEN = ROOT / "registrations" / "R24_multiwoz_random_negatives.frozen.json"
ON_PATH = ["scripts/knob_run.py", "scripts/newtable_run.py", "scripts/table_run.py", "scripts/abcd_confirm.py", "scripts/report_abcd.py",
           "scripts/seq_queue.sh", "src/dl/count_text.py", "src/dl/static_text.py", "src/dl/abcd.py", "src/dl/tm.py", "src/dl/tm_init.py",
           "src/dl/boolean.py", "src/dl/linear.py"]
ALLOWED: set = set()  # nothing, this checker included
DEVSTAGE = "324f6d626675c5a3600125561e12784d8b8df19519893644a74b4f3cc5ebe421"  # data/emb/multiwoz__bge-large.npz at the freeze
META = ("y", "convo_id", "k", "w_bert", "w_roberta")
VB, BGE = ["mem", "words", "xcust", "xrecent"], ["mem", "intent", "step", "words"]
ARMS = {"ref": {"tm": "tm_mem_ref_nrand", "lr": "lr_mem_ref", "encoder": "none", "parts": VB, "dev_row": "e83_multiwoz_a3B_c16000_s8_nrand", "e85": "refined", "short": "a3B",
                "table": "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855", "lr_dev": "dry_e82mwoz_lr_mem_a3B"},
        "c4": {"tm": "tm_mem_c4_nrand", "lr": "lr_mem_c4", "encoder": "none", "parts": VB, "dev_row": "e84_multiwoz_C4_c16000_s8_nrand", "e85": "C4", "short": "C4",
               "table": "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a", "lr_dev": "dry_e77mwoz_lr_mem_c4"},
        "bge": {"tm": "tm_mem_bge_nrand", "lr": "lr_mem_bge", "encoder": "bge-large", "parts": BGE, "dev_row": "e84_multiwoz_bge_c16000_s8_nrand", "e85": "bge-large", "short": "bge",
                "table": None, "lr_dev": "dry_e81multiwoz_lr_bge"}}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
utc = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))  # noqa: E731


def need(cond: bool, msg: str) -> None:
    if not cond:
        print("FAIL " + msg, flush=True)
        sys.exit(1)
    print("ok  " + msg, flush=True)


def sstar() -> dict:
    """Each agent's s* from e85's final output, and its 16k seed-1 dev row at s* (e83b / e84 at s 8, else e85b's)."""
    m = re.search(r"^s\*: (.+)$", (ROOT / "logs" / "e85_pick_final.out").read_text(), re.M)
    got = dict((a.rsplit(" ", 1)[0], int(a.rsplit(" ", 1)[1])) for a in m[1].split(", ")) if m else {}
    need(set(got) == {a["e85"] for a in ARMS.values()} and set(got.values()) <= {8, 12, 16, 24},
         "s* from logs/e85_pick_final.out: " + ", ".join(f"{a} s {s}" for a, s in got.items()))
    for a in ARMS.values():
        a["s"] = got[a["e85"]]
        if a["s"] != 8:
            a["dev_row"] = f"e85_multiwoz_{a['short']}_c16000_s{a['s']}_nrand"


def dev_rows() -> dict:
    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("seed") == 1 and d.get("tag") in {a["dev_row"] for a in ARMS.values()}:
            rows.setdefault(d["tag"], []).append(d)
    for a in ARMS.values():
        need(len(rows.get(a["dev_row"], [])) == 1, f"dev row {a['dev_row']} seed 1: exactly one")
    return {t: v[0] for t, v in rows.items()}


def check_tm(j: dict, a: dict, dev: dict, tag: str, clauses: int, epochs: int) -> None:
    need(j["arm"] == "tm" and j["parts"] == a["parts"] and j["encoder"] == a["encoder"] and j["cfg"]["neg_sampling"] == "random",
         f"{tag}: encoder {a['encoder']}, parts, random negatives")
    need(j.get("init") == dev.get("init") == "half", f"{tag}: init half, as {a['dev_row']}")
    if a["table"]:
        need(all(v["table_file_sha256"] == a["table"] for v in j["static_info"].values()), f"{tag}: the arm's table ({a['table'][:8]})")
    want = {**dev["cfg"], "seed": j["seed"], "n_clauses": clauses, "T": clauses, "epochs": epochs}
    need(j["cfg"] == want, f"{tag}: cfg equals {a['dev_row']}'s but the seed" + ("" if clauses == 16000 else f" and the dry size ({clauses}, {epochs})"))


def gate() -> None:
    """G-s* (dev, logs/plan_e77_note_2026-10-20.md's gate at each agent's e85 s*, seed 1, 16k): each network-free agent
    minus bge-large is at least -1.0. Exit 3 if both agents miss it (R24 stops; the test split stays unread)."""
    sstar()
    dev, b = dev_rows(), ARMS["bge"]
    held = []
    for k in ("c4", "ref"):
        a = ARMS[k]
        gap = 100 * (dev[a["dev_row"]]["acc"] - dev[b["dev_row"]]["acc"])
        held += [k] if gap >= -1.0 - 1e-9 else []
        print(f"G-s* {a['e85']}: {a['dev_row']} {dev[a['dev_row']]['acc']:.4f} minus {b['dev_row']} {dev[b['dev_row']]['acc']:.4f}: {gap:+.2f} "
              f"(needs >= -1.0): " + ("holds; its rules stand" if k in held else "missed; its arm runs with no rule ('gate missed at s*')"), flush=True)
    if not held:
        print("G-s*: both agents missed; R24 stops and MultiWOZ test stays unread", flush=True)
        sys.exit(3)
    print("G-s* done: " + ", ".join(ARMS[k]["tm"] for k in held) + " keep their rules", flush=True)


def main() -> None:
    if "--gate" in sys.argv[1:]:
        gate()
        return
    dry = "--dry" in sys.argv[1:]
    dev = dev_rows() if dry else None
    if dry:
        for k, a in ARMS.items():
            j = json.loads((DRY / f"dry_{REG}_{a['tm']}_s8" / "seed1.json").read_text())
            check_tm(j, a, dev[a["dev_row"]], f"dry {a['tm']}_s8", 200, 2)
            b, r = j["bits_sha256"], dev[a["dev_row"]]["bits_sha256"]
            need(b["train"] == r["train"] and b["dev"] == r["dev"] and b["test"] == b["dev"], f"dry {a['tm']}: train/dev bits equal {a['dev_row']}'s; test = dev")
            jl = json.loads((DRY / f"dry_{REG}_{a['lr']}" / "seed0.json").read_text())
            ld = json.loads((DRY / a["lr_dev"] / "seed0.json").read_text())
            need(jl["arm"] == "lr" and jl["bits_sha256"]["dev"] == r["dev"] and jl["dev_acc"] == ld["dev_acc"] and jl["C"] == ld["C"],
                 f"dry {a['lr']}: dev accuracy {jl['dev_acc']:.6f} and C {jl['C']} equal {a['lr_dev']}'s")
        print("all dry checks passed", flush=True)
        return
    sstar()
    dev = dev_rows()
    rec = json.loads(FROZEN.read_text())
    t_freeze = utc(rec["frozen_utc"])
    need(sha(ROOT / rec["registration"]) == rec["sha256"], f"registration text unchanged ({rec['sha256'][:16]})")
    now = {str(p.relative_to(ROOT)): sha(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
    for f in ON_PATH:
        need(now.get(f) == rec["sources"].get(f) is not None, f"on R24's path, unchanged: {f} ({now[f][:8]})")
    changed = sorted(f for f in set(now) | set(rec["sources"]) if rec["sources"].get(f) != now.get(f))
    need(set(changed) <= ALLOWED, f"other changes since the freeze: {changed}")
    jobs, lines = set(), []
    for line in (ROOT / "logs" / "jobs_R24.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            lines.append(line)
            jobs.add((re.findall(r"--name (\S+)", line)[-1], int(re.search(r"--seeds (\S+)", line)[1])))
    need(lines == r24_jobs({a["e85"]: a["s"] for a in ARMS.values()}), "logs/jobs_R24.txt is e85_pick.py's r24_jobs at s*, line for line")
    names = {a[x] for a in ARMS.values() for x in ("tm", "lr")}
    found = {(p.parent.name, int(p.stem[4:])) for n in names for p in (CONF / REG / n).glob("seed*.json")}
    need(len(jobs) == 18 and found == jobs, f"seed files are exactly the {len(jobs)} jobs of logs/jobs_R24.txt")
    ex = ROOT / "data" / "multiwoz" / "examples_test.json.gz"
    need(ex.stat().st_mtime > t_freeze, "data/multiwoz/examples_test.json.gz written after the freeze")
    with gzip.open(ex, "rt") as f, gzip.open(ROOT / "data" / "multiwoz" / "examples_train.json.gz", "rt") as g:
        te = json.load(f)
        need(te["buttons"] == json.load(g)["buttons"], "examples_test uses train's label list")
    n_turns, n_dlg = len(te["rows"]), len({r["convo_id"] for r in te["rows"]})
    dv = ROOT / "data" / "emb" / "multiwoz__bge-large.devstage.npz"
    need(sha(dv) == DEVSTAGE, f"the bge-large dev-stage file has its frozen hash ({DEVSTAGE[:8]})")
    e, s0 = np.load(ROOT / "data" / "emb" / "multiwoz__bge-large.npz"), np.load(dv)
    ks = [k for k in s0.files if k != "provenance"]
    need(all(np.array_equal(e[k], s0[k]) for k in ks), f"the bge-large file's {len(ks)} train/dev arrays equal the dev-stage file's (provenance aside)")
    meta0, test_bits = None, {}
    for k, a in ARMS.items():
        d = dev[a["dev_row"]]
        for seed in range(1, 6):
            folder, tag = CONF / REG / a["tm"], f"{REG}/{a['tm']}/seed{seed}"
            j = json.loads((folder / f"seed{seed}.json").read_text())
            need(j["registration"] == REG and j["name"] == a["tm"] and j["seed"] == seed, f"{tag}: registration, name, seed")
            check_tm(j, a, d, tag, 16000, 40)
            need(utc(j["time"]) > t_freeze and min(p.stat().st_mtime for p in folder.iterdir()) > t_freeze, f"{tag}: written {j['time']}, after the freeze")
            dj = json.loads((DRY / f"dry_{REG}_{a['tm']}_s8" / "seed1.json").read_text())
            need(j["bits_sha256"]["train"] == dj["bits_sha256"]["train"] == d["bits_sha256"]["train"]
                 and j["bits_sha256"]["dev"] == dj["bits_sha256"]["dev"] == d["bits_sha256"]["dev"], f"{tag}: train/dev bits equal the dry run's and the dev row's")
            if seed == 1:
                need(j["dev_acc"] == d["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals {a['dev_row']}'s exactly")
            z = np.load(folder / f"seed{seed}.npz")
            need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy")
            meta0 = meta0 or {m: z[m] for m in META}
            need(all(np.array_equal(z[m], meta0[m]) for m in META), f"{tag}: test rows equal the other arms' ({', '.join(META)})")
            test_bits.setdefault(k, set()).add(j["bits_sha256"]["test"])
        need(len(test_bits[k]) == 1, f"{a['tm']}: one test-bit hash over seeds ({next(iter(test_bits[k]))[:8]})")
        jl = json.loads((CONF / REG / a["lr"] / "seed0.json").read_text())
        ld = json.loads((DRY / a["lr_dev"] / "seed0.json").read_text())
        need(jl["arm"] == "lr" and jl["bits_sha256"]["test"] in test_bits[k] and jl["bits_sha256"]["dev"] == d["bits_sha256"]["dev"]
             and jl["dev_acc"] == ld["dev_acc"] and jl["C"] == ld["C"], f"{a['lr']}: bits of its TM arm; dev accuracy and C ({jl['C']}) equal {a['lr_dev']}'s")
        z = np.load(CONF / REG / a["lr"] / "seed0.npz")
        need(all(np.array_equal(z[m], meta0[m]) for m in META), f"{a['lr']}: the same test rows ({', '.join(META)})")
    n_seen = len(np.unique(meta0["convo_id"]))
    need(len(meta0["y"]) == n_turns and n_seen == n_dlg, f"{len(meta0['y'])} test turns in {n_seen} dialogues, as in examples_test.json.gz ({n_turns} in {n_dlg})")
    rows = [json.loads(x) for x in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines()]
    e85 = sorted(d["time"] for d in rows if (d.get("tag") or "").startswith("e85_"))
    early = [t for t in e85 if utc(t) <= t_freeze]
    print(f"note: {len(early)} of {len(e85)} e85 rows were written before the freeze" + (f" ({early[0]} to {early[-1]})" if early else ""), flush=True)
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
