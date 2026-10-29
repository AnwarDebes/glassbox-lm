#!/usr/bin/env python
"""W1-T verdict addendum (the two conditions of 04:26Z; prints, decides nothing):
1. the reference arms are N7's registered test rows: file hashes, the code hashes stored in those rows, and the seeds;
2. drift check: the word arm's carved-val S1 per seed in the test runs next to the W1-S val rows of the same table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMB = HERE.parent


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    n7 = EMB / "results_n7_test.jsonl"
    print(f"reference arms: N7's registered test rows, {n7.name} sha256 {sha16(n7)}; REGISTRATION_N7.md {sha16(EMB / 'REGISTRATION_N7.md')}; "
          f"verdict_n7.txt {sha16(EMB / 'verdict_n7.txt')}")
    ref = [json.loads(x) for x in n7.read_text().splitlines() if x.strip()]
    ref = [r for r in ref if r["view"] in ("existsT_potion", "existsT_glove", "bow")]
    hs = sorted({json.dumps(r["sha256"], sort_keys=True) for r in ref})
    print(f"   {len(ref)} rows (views existsT_potion, existsT_glove, bow); code hashes stored in them: {hs}")
    seeds = {}
    for r in ref:
        seeds.setdefault(f"{r['dataset']}/{r['view']}", []).append(int(r["seed"]))
    print(f"   seeds per set and view: {{{', '.join(f'{k}: {sorted(s)}' for k, s in sorted(seeds.items()))}}}")
    w = [json.loads(x) for x in (HERE / "results_w1t_test.jsonl").read_text().splitlines() if x.strip()]
    v = [json.loads(x) for x in (HERE / "results_w1_n7val.jsonl").read_text().splitlines() if x.strip()]
    v = {(r["dataset"], r["seed"]): r for r in v if r["table"] == "ppw_S1000_d500_e0.5"}
    print("drift check: carved-val S1 (mean val accuracy, epochs 51-100) of the word arm, test run vs W1-S val run")
    for ds in ("trec", "mr", "sst2", "r52", "r8"):
        line = f"   {ds}:"
        for r in sorted((x for x in w if x["dataset"] == ds), key=lambda x: x["seed"]):
            old = v.get((ds, r["seed"]))
            line += f" s{r['seed']} {r['val_mean_last50']:.3f}" + (f" (W1-S {old['val_S1_mean_51_100']:.3f}, diff {r['val_mean_last50'] - old['val_S1_mean_51_100']:+.3f})" if old else " (no W1-S row)")
        print(line)
    same = all(json.dumps(r["cache_meta"]["train_val_bits_equal_val_run_cache"]) == '{"train": true, "val": true}' for r in w)
    print(f"train and val bits equal the W1-S val-run cache in every test row: {same}")


if __name__ == "__main__":
    main()
