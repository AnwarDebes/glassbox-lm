#!/usr/bin/env python
"""e84 (plan logs/plan_e84_negatives_fair.md): the three MultiWOZ agents with random negatives, and the R24 gate
re-check. Read only. Written before any e84 number.

    e84_check.py    # after jobs_e84.txt

The rows: the bge-large and C4 agents' e84 rows (random negatives, 16k, T 16,000, s 8, seeds 1-2), beside the refined
agent's e83b rows (the same settings).
The gates (fixed in the plan), each on the mean of seeds 1-2:
- G-ref: refined minus bge-large is at least -1.0 at best-dev;
- G-C4: C4 minus bge-large is at least -1.0 at best-dev.
A gate is steady if it also holds on the last-10 mean. A gate met at best-dev only is written "met on best-dev only"
and allows no R24 draft.
Checks on every row:
- the agent's encoder, parts, dev bits and table;
- 16k, T 16,000, s 8, 40 epochs, lex_k 1000, start half, neg_sampling random, otherwise e83's BASE_CFG;
- e84 rows written after e84's freeze and e83 rows after e83's;
- exactly one row per tag and seed;
- the preds file: MultiWOZ dev's labels, the evaluation counts of the history, and the row's acc from the final
  prediction.
Reported beside the gates (no rule): each agent's random minus hardest negatives, and TM minus LR on its own bits.
Tested by tests/test_e84_check.py (made-up rows).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VB = ["mem", "words", "xcust", "xrecent"]
AG = {
    "refined": {"enc": "none", "parts": VB, "dev": "612b8f0b75e32a30b50493eaedc1eec9d6bac6893d48075144a1f48fad2e23a3",
                "table": "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855", "tag": "e83_multiwoz_a3B_c16000_s8_nrand",
                "preds": "e83_preds", "rec": "plan_e83_learning_dynamics", "lr": "dry_e82mwoz_lr_mem_a3B",
                "hard": [("e82_multiwoz_Bt_c16000_s8", 1), ("e82_multiwoz_Bt_c16000_s8", 2)]},
    "bge-large": {"enc": "bge-large", "parts": ["mem", "intent", "step", "words"], "dev": "b068b64787582fc8c00dd8a446cb415fcb9737d3069e7111f092503bf7bf2797",
                  "table": None, "tag": "e84_multiwoz_bge_c16000_s8_nrand", "preds": "e84_preds", "rec": "plan_e84_negatives_fair",
                  "lr": "dry_e81multiwoz_lr_bge", "hard": [("e77_mwoz_bge_c16000_s8", 1)]},
    "C4": {"enc": "none", "parts": VB, "dev": "311e8933d2724bf8fc3ee1a5cef412010bd3a580c076217e3a964bcd286d68ba",
           "table": "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a", "tag": "e84_multiwoz_C4_c16000_s8_nrand",
           "preds": "e84_preds", "rec": "plan_e84_negatives_fair", "lr": "dry_e77mwoz_lr_mem_c4",
           "hard": [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)]},
}
BASE_CFG = {"n_states": 256, "epochs": 40, "batch_size": 128, "neg_sampling": "hardest", "negation": True, "max_literals": 0, "budget_mode": "exact",
            "max_negated": 0, "budget_select": "rank", "clause_drop": 0.0, "half": True, "eval_every": 2}


def stats(d: dict) -> tuple[float, float, float]:
    v = [e["val_acc"] for e in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / len(v[-10:])


def main() -> None:
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)
    with gzip.open(ROOT / "data" / "multiwoz" / "examples_dev.json.gz", "rt") as f:
        y = np.array([r["y"] for r in json.load(f)["rows"]])
    frozen = {a: json.loads((ROOT / "logs" / f"{g['rec']}.frozen.json").read_text())["frozen_utc"] for a, g in AG.items()}

    def get(agent: str, t: str, seed: int, neg: str) -> dict:
        got = rows.get((t, seed), [])
        if len(got) != 1:
            raise SystemExit(f"{t} seed {seed}: {len(got)} rows (need exactly 1)")
        d, g = got[0], AG[agent]
        want = {**BASE_CFG, "neg_sampling": neg}
        if not (d["dataset"] == "multiwoz" and d["encoder"] == g["enc"] and list(d["parts"]) == g["parts"] and d.get("lex_k") == 1000
                and d.get("init") == "half" and (d["cfg"]["n_clauses"], d["cfg"]["T"], d["cfg"]["s"], d["cfg"]["seed"]) == (16000, 16000, 8.0, seed)
                and {k: v for k, v in d["cfg"].items() if k not in ("n_clauses", "T", "s", "seed")} == want):
            raise SystemExit(f"{t} seed {seed}: settings differ from the plan's")
        if d["bits_sha256"]["dev"] != g["dev"]:
            raise SystemExit(f"{t} seed {seed}: dev bits {d['bits_sha256']['dev'][:8]} are not the {agent} agent's")
        if g["table"] and not all(d["static_info"][k]["table_file_sha256"] == g["table"] for k in ("xcust", "xrecent")):
            raise SystemExit(f"{t} seed {seed}: not on the {agent} agent's table")
        if t == g["tag"]:
            if d["time"] <= frozen[agent]:
                raise SystemExit(f"{t} seed {seed}: written before its plan's freeze")
            p = ROOT / "results" / "explore" / g["preds"] / f"{t}_seed{seed}.npz"
            if not p.exists():
                raise SystemExit(f"{t} seed {seed}: no preds file")
            with np.load(p) as z:
                P, yv = z["preds"].astype(np.int64), z["y_val"].astype(np.int64)
            n, h = len(y), d["history"]
            if not np.array_equal(yv, y) or P.shape != (len(h) + 1, n):
                raise SystemExit(f"{t} seed {seed}: preds file does not fit MultiWOZ dev or the history")
            if any(int((P[k] == y).sum()) != round(e["val_acc"] * n) for k, e in enumerate(h)) or float((P[-1] == y).mean()) != d["acc"]:
                raise SystemExit(f"{t} seed {seed}: preds file does not give the history or the row's acc")
        return d

    rnd, hard, lr = {}, {}, {}
    for a, g in AG.items():
        s = [stats(get(a, g["tag"], sd, "random")) for sd in (1, 2)]
        rnd[a] = [(s[0][i] + s[1][i]) / 2 for i in range(3)]
        h = [stats(get(a, t, sd, "hardest")) for t, sd in g["hard"]]
        hard[a] = ([sum(x[i] for x in h) / len(h) for i in range(3)], len(h))
        j = json.loads((ROOT / "results" / "explore" / "dryrun" / g["lr"] / "seed0.json").read_text())
        if j["arm"] != "lr" or j["bits_sha256"]["dev"] != g["dev"]:
            raise SystemExit(f"{g['lr']}: not LR on the {a} agent's dev bits")
        lr[a] = j["dev_acc"]
    print("random negatives, MultiWOZ dev, 16k, T 16000, s 8, mean of seeds 1-2: best-dev / final epoch / last-10 mean")
    for a in AG:
        hb, n = hard[a]
        print(f"  {a:9} {rnd[a][0]:.4f} / {rnd[a][1]:.4f} / {rnd[a][2]:.4f}   beside it (no rule): minus hardest ({n} seed(s)) best-dev "
              f"{100 * (rnd[a][0] - hb[0]):+.2f}, last-10 {100 * (rnd[a][2] - hb[2]):+.2f}; TM minus LR {100 * (rnd[a][0] - lr[a]):+.2f}")
    for name, a in (("G-ref", "refined"), ("G-C4", "C4")):
        gb, gl = 100 * (rnd[a][0] - rnd["bge-large"][0]), 100 * (rnd[a][2] - rnd["bge-large"][2])
        met, steady = gb >= -1.0 - 1e-9, gl >= -1.0 - 1e-9
        word = "MET, steady" if met and steady else ("met on best-dev only" if met else "not met")
        print(f"{name} (R24 gate: {a} minus bge-large): best-dev {gb:+.2f} (needs >= -1.0), last-10 {gl:+.2f}: {word}"
              + ("; an R24 draft may follow (test unread until it is frozen)" if met and steady else "; no R24 draft"))


if __name__ == "__main__":
    main()
