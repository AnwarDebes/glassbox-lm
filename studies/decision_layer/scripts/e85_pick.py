#!/usr/bin/env python
"""e85 (plan logs/plan_e85_s_under_random.md): the TM's s for each MultiWOZ agent under random negatives, by e77's rule.
Picks, the follow-up jobs, and R24's job file. Read only, except that it writes job files, refusing to overwrite them.
Written before any e85 number.

    e85_pick.py --stage screen   # after jobs_e85a.txt: the 4k picks; writes logs/jobs_e85b.txt (each moved pick at 16k)
    e85_pick.py --stage final    # after jobs_e85b.txt: s* per agent; writes logs/jobs_R24.txt (R24's arms at s*)

Rules (e77's, fixed in the plan), for each agent:
- screen: random negatives, MultiWOZ dev, 4,000 clauses (T 4,000), seed 1, s in [8, 12, 16, 24]. The pick is the
  highest 4k dev score if it is at least 0.3 points above the agent's own s 8, otherwise s 8; an exact tie goes to the
  first of the list;
- final: s* is the pick, unless its 16k dev (seed 1) is below the agent's own s 8 at 16k (seed 1, the e83b / e84 row);
  then s 8.
Checks on every row: e84_check.py's (the agent's encoder, parts, dev bits and table; random negatives, start half,
40 epochs, lex_k 1000; e83's BASE_CFG otherwise), clauses, T and s as the tag says, e85 rows after the plan's freeze,
exactly one row per tag and seed, and each e85 row's preds file. Tested by tests/test_e85_pick.py (made-up rows).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e84_check import AG, BASE_CFG  # noqa: E402

S = [8, 12, 16, 24]
SHORT = {"refined": "a3B", "bge-large": "bge", "C4": "C4"}
TH = "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 .venv-dl/bin/python"
PRE = {"refined": "scripts/newtable_run.py --table-dir data/tables/c4words_a3B_v1 --name c4words-a3B-v1 {script} --encoder none --parts mem,words,xcust,xrecent",
       "C4": "scripts/table_run.py --table c4words-S1000-v1 {script} --encoder none --parts mem,words,xcust,xrecent",
       "bge-large": "{script} --encoder bge-large --parts mem,intent,step,words"}
R24_NAME = {"refined": "ref", "C4": "c4", "bge-large": "bge"}
GRID = "0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100"


def tag(agent: str, clauses: int, s: int) -> str:
    return f"e85_multiwoz_{SHORT[agent]}_c{clauses}_s{s}_nrand"


def job(agent: str, clauses: int, s: int, seed: int = 1) -> str:
    t = tag(agent, clauses, s)
    return (f"{TH} scripts/knob_run.py --neg-sampling random --preds-out results/explore/e85_preds/{t}_seed{seed}.npz -- "
            f"{PRE[agent].format(script='scripts/abcd_explore.py')} --arms tm --save --dataset multiwoz --clauses {clauses} --T {clauses} "
            f"--s {s} --seed {seed} --tag {t} >> logs/{t}_seed{seed}.log 2>&1")


def r24_jobs(sstar: dict) -> list[str]:
    out = []
    for agent in ("refined", "C4", "bge-large"):
        name = f"tm_mem_{R24_NAME[agent]}_nrand"
        for sd in range(1, 6):
            out.append(f"{TH} scripts/knob_run.py --neg-sampling random -- {PRE[agent].format(script='scripts/abcd_confirm.py')} --dataset multiwoz "
                       f"--registration R24mwoz --arm tm --s {sstar[agent]} --name {name} --clauses 16000 --seeds {sd} >> logs/R24.out 2>&1")
    for agent in ("refined", "C4", "bge-large"):
        out.append(f"{TH} {PRE[agent].format(script='scripts/abcd_confirm.py')} --dataset multiwoz --registration R24mwoz --arm lr "
                   f"--name lr_mem_{R24_NAME[agent]} --grid-c {GRID} --seeds 0 >> logs/R24.out 2>&1")
    return out


class Rows:
    def __init__(self) -> None:
        self.frozen = json.loads((ROOT / "logs" / "plan_e85_s_under_random.frozen.json").read_text())["frozen_utc"]
        self.r: dict = {}
        for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)
        with gzip.open(ROOT / "data" / "multiwoz" / "examples_dev.json.gz", "rt") as f:
            self.y = np.array([r["y"] for r in json.load(f)["rows"]])

    def get(self, agent: str, t: str, seed: int, clauses: int, s: int) -> dict:
        got = self.r.get((t, seed), [])
        if len(got) != 1:
            raise SystemExit(f"{t} seed {seed}: {len(got)} rows (need exactly 1)")
        d, g = got[0], AG[agent]
        want = {**BASE_CFG, "neg_sampling": "random"}
        if not (d["dataset"] == "multiwoz" and d["encoder"] == g["enc"] and list(d["parts"]) == g["parts"] and d.get("lex_k") == 1000
                and d.get("init") == "half" and (d["cfg"]["n_clauses"], d["cfg"]["T"], d["cfg"]["s"], d["cfg"]["seed"]) == (clauses, clauses, float(s), seed)
                and {k: v for k, v in d["cfg"].items() if k not in ("n_clauses", "T", "s", "seed")} == want):
            raise SystemExit(f"{t} seed {seed}: settings differ from the plan's")
        if d["bits_sha256"]["dev"] != g["dev"]:
            raise SystemExit(f"{t} seed {seed}: dev bits {d['bits_sha256']['dev'][:8]} are not the {agent} agent's")
        if g["table"] and not all(d["static_info"][k]["table_file_sha256"] == g["table"] for k in ("xcust", "xrecent")):
            raise SystemExit(f"{t} seed {seed}: not on the {agent} agent's table")
        if t.startswith("e85"):
            if d["time"] <= self.frozen:
                raise SystemExit(f"{t} seed {seed}: written before the freeze")
            p = ROOT / "results" / "explore" / "e85_preds" / f"{t}_seed{seed}.npz"
            if not p.exists():
                raise SystemExit(f"{t} seed {seed}: no preds file")
            with np.load(p) as z:
                P, yv = z["preds"].astype(np.int64), z["y_val"].astype(np.int64)
            n, h = len(self.y), d["history"]
            if not np.array_equal(yv, self.y) or P.shape != (len(h) + 1, n) or float((P[-1] == self.y).mean()) != d["acc"] \
                    or any(int((P[k] == self.y).sum()) != round(e["val_acc"] * n) for k, e in enumerate(h)):
                raise SystemExit(f"{t} seed {seed}: preds file does not fit MultiWOZ dev, the history or the row's acc")
        return d


def write(name: str, lines: list[str], what: str) -> None:
    out = ROOT / "logs" / name
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text("\n".join([f"# {name[:-4]} ({what}). Written by scripts/e85_pick.py. One at a time through seq_queue.sh."] + lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(lines)} jobs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen", "final"])
    args = ap.parse_args()
    R = Rows()
    picks = {}
    for agent in AG:
        acc = {s: R.get(agent, tag(agent, 4000, s), 1, 4000, s)["acc"] for s in S}
        best = max(S, key=lambda s: (acc[s], -S.index(s)))
        picks[agent] = best if acc[best] - acc[8] >= 0.003 - 1e-12 else 8
        print(f"{agent:9} 4k seed 1: " + ", ".join(f"s {s} {acc[s]:.4f}" for s in S) + f"; best s {best}, {100 * (acc[best] - acc[8]):+.2f} over s 8; pick s {picks[agent]}")
    if args.stage == "screen":
        write("jobs_e85b.txt", [job(a, 16000, p) for a, p in picks.items() if p != 8], "plan logs/plan_e85_s_under_random.md: each moved pick at 16k, seed 1")
        return
    sstar = {}
    for agent, p in picks.items():
        a8 = R.get(agent, AG[agent]["tag"], 1, 16000, 8)["acc"]  # the agent's own s 8 at 16k, seed 1 (e83b / e84)
        if p == 8:
            sstar[agent] = 8
            print(f"{agent:9}: pick s 8 -> s* 8 (16k s 8 {a8:.4f})")
            continue
        ap_ = R.get(agent, tag(agent, 16000, p), 1, 16000, p)["acc"]
        sstar[agent] = 8 if ap_ < a8 else p
        print(f"{agent:9}: 16k s 8 {a8:.4f}; 16k pick s {p} {ap_:.4f} -> s* {sstar[agent]}")
    print("s*: " + ", ".join(f"{a} {s}" for a, s in sstar.items()))
    write("jobs_R24.txt", r24_jobs(sstar), "registrations/R24_multiwoz_random_negatives.md: MultiWOZ test, 16k, random negatives, each agent at its e85 s*, seeds 1-5, then LR")


if __name__ == "__main__":
    main()
