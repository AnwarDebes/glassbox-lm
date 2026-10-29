#!/usr/bin/env python
"""e86 (plan logs/plan_e86_clause_drop.md): clause dropout 0.5 for the C4 agent on MultiWOZ, STAR and ABCD dev, and
random negatives with dropout as one setting for all three. Read only. Written before any e86 number.

    e86_check.py          # after jobs_e86.txt
    e86_check.py --dry    # after jobs_e86_dry.txt (tiny TM: the knobs, bits, table and preds files, no rule)

The C4 agent (no neural network at any stage): table c4words_S1000_v1, view mem, words, xcust, xrecent, 40 epochs,
start half, seeds 1-2. Each dataset keeps its size and s from earlier dev work: MultiWOZ 16k, T 16,000, s 8 (e85);
STAR 16k, T 16,000, s 16 (e74c, R22); ABCD 4k, T 4,000, s 16 (e75, R23). The cells:
- MultiWOZ: random negatives + drop 0.5 (e86), against random negatives, no drop (e84's rows);
- STAR and ABCD: hardest + drop 0.5 and random + drop 0.5 (e86), against hardest, no drop (e86's fresh rows).
The rules (fixed in the plan), on the mean of seeds 1-2, in points:
- per dataset, drop 0.5 is kept if the dropout cell minus the reference is at least +0.5 at best-dev and on the
  last-10 mean (the dropout cell: random + drop on MultiWOZ, hardest + drop on STAR and ABCD);
- E86.1: drop 0.5 kept on all three datasets;
- E86.3: random + drop 0.5 minus each dataset's best setting after the keep rule (its dropout cell where dropout is
  kept, else its reference) is at least -0.5 at best-dev and on the last-10 mean, on all three datasets (one
  negative-sampling setting).
Reported with no rule: every seed, the final epoch, TM minus LR on the same bits (the existing LR dev rows), and the
fresh STAR and ABCD references' seed 1 beside R22's and R23's seed-1 dev rows.
Checks on every row: dataset, encoder, parts, dev bits, table, lex_k 1000, start half, size, T, s, seed, neg_sampling
and clause_drop as the cell's, the rest of the cfg as e84's; e86 rows written after e86's freeze and e84 rows after
e84's; exactly one row per tag and seed; the preds file gives the dev labels, the history and the row's acc.
Tested by tests/test_e86_check.py (made-up rows).

Amendment of 2026-10-25 (dated note logs/plan_e86_note_2026-10-25.md, frozen before the six STAR jobs ran again):
the e86 queue was stopped at 2026-10-23T02:26:38Z during its first STAR job, after that job had written its row and
before it wrote its preds file. The six STAR jobs ran again after the note's freeze. For a STAR tag and seed, a row
written at or before the note's freeze is set aside and printed (no rule), and the check needs exactly one row written
after it. The note's freeze record must exist for the full check. Nothing else changes.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VB = ["mem", "words", "xcust", "xrecent"]
TABLE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
BASE_CFG = {"n_states": 256, "epochs": 40, "batch_size": 128, "neg_sampling": "hardest", "negation": True, "max_literals": 0, "budget_mode": "exact",
            "max_negated": 0, "budget_select": "rank", "clause_drop": 0.0, "half": True, "eval_every": 2}
E86, E84 = "plan_e86_clause_drop", "plan_e84_negatives_fair"
NOTE = "plan_e86_note_2026-10-25"  # amendment: the STAR jobs ran again after the stop of 2026-10-23T02:26:38Z
SET_ASIDE: list = []
DS = {
    "multiwoz": {"clauses": 16000, "s": 8.0, "dev": "311e8933d2724bf8fc3ee1a5cef412010bd3a580c076217e3a964bcd286d68ba",
                 "lr": "dry_e77mwoz_lr_mem_c4", "earlier": None},
    "star": {"clauses": 16000, "s": 16.0, "dev": "3fc00b173cf402a56f152dfcea2a5f9e26b02a2d2100ba7dec76fa063851d4f7",
             "lr": "dry_R21star_lr_mem_c4", "earlier": ("R22", "results/confirm/R22star/tm_mem_c4_s16/seed1.json")},
    "abcd": {"clauses": 4000, "s": 16.0, "dev": "56297e10edd0532f15823b602508976eaa28fbeb5fd0f14893eb862620014d33",
             "lr": "dry_R21abcd_lr_mem_c4", "earlier": ("R23", "results/confirm/R23abcd/tm_mem_c4_s16/seed1.json")},
}
# per dataset: (cell, tag, neg_sampling, clause_drop, preds folder, freeze record)
CELLS = {
    "multiwoz": [("ref", "e84_multiwoz_C4_c16000_s8_nrand", "random", 0.0, "e84_preds", E84),
                 ("nrand_drop50", "e86_multiwoz_C4_c16000_s8_nrand_drop50", "random", 0.5, "e86_preds", E86)],
    "star": [("ref", "e86_star_C4_c16000_s16_ref", "hardest", 0.0, "e86_preds", E86),
             ("drop50", "e86_star_C4_c16000_s16_drop50", "hardest", 0.5, "e86_preds", E86),
             ("nrand_drop50", "e86_star_C4_c16000_s16_nrand_drop50", "random", 0.5, "e86_preds", E86)],
    "abcd": [("ref", "e86_abcd_C4_c4000_s16_ref", "hardest", 0.0, "e86_preds", E86),
             ("drop50", "e86_abcd_C4_c4000_s16_drop50", "hardest", 0.5, "e86_preds", E86),
             ("nrand_drop50", "e86_abcd_C4_c4000_s16_nrand_drop50", "random", 0.5, "e86_preds", E86)],
}
DROP_CELL = {"multiwoz": "nrand_drop50", "star": "drop50", "abcd": "drop50"}
NAME = {"ref": "reference", "drop50": "hardest + drop 0.5", "nrand_drop50": "random + drop 0.5"}
SEEDS = (1, 2)
DRY = {"clauses": 200, "T": 50, "epochs": 2}


def stats(d: dict) -> tuple[float, float, float]:
    v = [e["val_acc"] for e in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / len(v[-10:])


def load_rows() -> dict:
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)
    return rows


def labels(ds: str) -> np.ndarray:
    with gzip.open(ROOT / "data" / ds / "examples_dev.json.gz", "rt") as f:
        return np.array([r["y"] for r in json.load(f)["rows"]])


def get(rows: dict, frozen: dict, y: np.ndarray, ds: str, tag: str, seed: int, neg: str, drop: float, preds: str, rec: str,
        dry: bool = False) -> dict:
    got = rows.get((tag, seed), [])
    if not dry and ds == "star":
        SET_ASIDE.extend((tag, seed, r["time"], r["acc"]) for r in got if r["time"] <= frozen[NOTE])
        got = [r for r in got if r["time"] > frozen[NOTE]]
    if len(got) != 1:
        raise SystemExit(f"{tag} seed {seed}: {len(got)} rows (need exactly 1)")
    d, g = got[0], DS[ds]
    n_c, T = (DRY["clauses"], DRY["T"]) if dry else (g["clauses"], g["clauses"])
    want = {**BASE_CFG, "neg_sampling": neg, "clause_drop": drop, **({"epochs": DRY["epochs"]} if dry else {})}
    if not (d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == VB and d.get("lex_k") == 1000 and d.get("init") == "half"
            and (d["cfg"]["n_clauses"], d["cfg"]["T"], d["cfg"]["s"], d["cfg"]["seed"]) == (n_c, T, g["s"], seed)
            and {k: v for k, v in d["cfg"].items() if k not in ("n_clauses", "T", "s", "seed")} == want):
        raise SystemExit(f"{tag} seed {seed}: settings differ from the plan's")
    if d["bits_sha256"]["dev"] != g["dev"]:
        raise SystemExit(f"{tag} seed {seed}: dev bits {d['bits_sha256']['dev'][:8]} are not the C4 agent's on {ds}")
    if not all(d["static_info"][k]["table_file_sha256"] == TABLE for k in ("xcust", "xrecent")):
        raise SystemExit(f"{tag} seed {seed}: not on the C4 table")
    if d["time"] <= frozen[rec]:
        raise SystemExit(f"{tag} seed {seed}: written before its plan's freeze")
    p = ROOT / "results" / "explore" / preds / f"{tag}_seed{seed}.npz"
    if not p.exists():
        raise SystemExit(f"{tag} seed {seed}: no preds file")
    with np.load(p) as z:
        P, yv = z["preds"].astype(np.int64), z["y_val"].astype(np.int64)
    n, h = len(y), d["history"]
    if not np.array_equal(yv, y) or P.shape != (len(h) + 1, n):
        raise SystemExit(f"{tag} seed {seed}: preds file does not fit {ds} dev or the history")
    if any(int((P[k] == y).sum()) != round(e["val_acc"] * n) for k, e in enumerate(h)) or float((P[-1] == y).mean()) != d["acc"]:
        raise SystemExit(f"{tag} seed {seed}: preds file does not give the history or the row's acc")
    return d


def main() -> None:
    dry = sys.argv[1:] == ["--dry"]
    if sys.argv[1:] not in ([], ["--dry"]):
        raise SystemExit(__doc__)
    rows = load_rows()
    frozen = {r: json.loads((ROOT / "logs" / f"{r}.frozen.json").read_text())["frozen_utc"] for r in (E86, E84)}
    if dry:
        for ds, cells in CELLS.items():
            y = labels(ds)
            for cell, tag, neg, drop, preds, rec in cells:
                if rec != E86:
                    continue
                t = f"e86dry_{ds}_{cell}"
                d = get(rows, frozen, y, ds, t, 1, neg, drop, preds, rec, dry=True)
                print(f"ok {t} seed 1: knobs neg_sampling {neg}, clause_drop {drop}; dev bits {d['bits_sha256']['dev'][:8]}; C4 table; preds file fits")
        print("all dry checks passed")
        return
    note = ROOT / "logs" / f"{NOTE}.frozen.json"
    if not note.exists():
        raise SystemExit(f"{note.name} is missing: the STAR jobs of e86 run only after the dated note of 2026-10-25 is frozen")
    frozen[NOTE] = json.loads(note.read_text())["frozen_utc"]
    res, lr = {}, {}
    for ds, cells in CELLS.items():
        y, g = labels(ds), DS[ds]
        j = json.loads((ROOT / "results" / "explore" / "dryrun" / g["lr"] / "seed0.json").read_text())
        if j["arm"] != "lr" or j["bits_sha256"]["dev"] != g["dev"]:
            raise SystemExit(f"{g['lr']}: not LR on the C4 agent's {ds} dev bits")
        lr[ds] = j["dev_acc"]
        for cell, tag, neg, drop, preds, rec in cells:
            per = [stats(get(rows, frozen, y, ds, tag, sd, neg, drop, preds, rec)) for sd in SEEDS]
            res[ds, cell] = ([sum(x[i] for x in per) / len(per) for i in range(3)], per)
    print("e86, the C4 agent (no neural network at any stage), dev, mean of seeds 1-2: best-dev / final epoch / last-10 mean;"
          " per seed best-dev; TM minus LR on the same bits (no rule)")
    for ds, cells in CELLS.items():
        g = DS[ds]
        print(f"  {ds} ({g['clauses'] // 1000}k, T {g['clauses']}, s {g['s']:g}; LR dev {lr[ds]:.4f} from {g['lr']}):")
        for cell, tag, neg, drop, *_ in cells:
            m, per = res[ds, cell]
            print(f"    {NAME[cell]:19} ({neg}, drop {drop:g}) {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f}   seeds "
                  + ", ".join(f"{x[0]:.4f}" for x in per) + f"   TM minus LR {100 * (m[0] - lr[ds]):+.2f}")
        if g["earlier"]:
            reg, path = g["earlier"]
            e = json.loads((ROOT / path).read_text())
            if e["bits_sha256"]["dev"] != g["dev"] or e["cfg"]["s"] != g["s"] or e["cfg"]["n_clauses"] != g["clauses"] or e["cfg"]["seed"] != 1:
                raise SystemExit(f"{path}: not the C4 agent's seed-1 row at this size and s")
            print(f"    beside it (no rule): the reference's seed 1 minus {reg}'s seed-1 dev row ({e['dev_acc']:.4f}): "
                  f"{100 * (res[ds, 'ref'][1][0][0] - e['dev_acc']):+.2f}")
    kept = []
    print("Dropout per dataset (the dropout cell minus the reference; needs >= +0.5 at best-dev and on the last-10 mean):")
    for ds in CELLS:
        a, r = res[ds, DROP_CELL[ds]][0], res[ds, "ref"][0]
        b, l10 = 100 * (a[0] - r[0]), 100 * (a[2] - r[2])
        ok = b >= 0.5 - 1e-9 and l10 >= 0.5 - 1e-9
        kept += [ds] if ok else []
        print(f"  {ds}: {NAME[DROP_CELL[ds]]} best-dev {b:+.2f}, last-10 {l10:+.2f}: " + ("kept" if ok else "not kept"))
    print("E86.1 (drop 0.5 kept on all three datasets): " + ("MET" if len(kept) == 3 else "not met")
          + f" (kept on: {', '.join(kept) if kept else 'none'})")
    one = []
    for ds in CELLS:
        best = DROP_CELL[ds] if ds in kept else "ref"
        a, r = res[ds, "nrand_drop50"][0], res[ds, best][0]
        b, l10 = 100 * (a[0] - r[0]), 100 * (a[2] - r[2])
        one.append(b >= -0.5 - 1e-9 and l10 >= -0.5 - 1e-9)
        print(f"  E86.3 {ds}: random + drop 0.5 minus the dataset's best setting ({NAME[best]}) best-dev {b:+.2f}, last-10 {l10:+.2f}"
              " (needs >= -0.5): " + ("holds" if one[-1] else "fails"))
    print("E86.3 (random + drop 0.5 as one setting for all three datasets): " + ("MET" if all(one) else "not met"))
    for tag, seed, when, acc in SET_ASIDE:
        after = [r for r in rows.get((tag, seed), []) if r["time"] > frozen[NOTE]]
        print(f"Set aside (no rule; the queue stopped on 2026-10-23): {tag} seed {seed}, written {when}, acc {acc:.4f};"
              f" the row written after the note: {after[0]['acc']:.4f}, difference {100 * (after[0]['acc'] - acc):+.2f}")


if __name__ == "__main__":
    main()
