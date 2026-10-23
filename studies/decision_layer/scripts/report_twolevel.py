#!/usr/bin/env python
"""A two-level bootstrap interval for registered pairs (no rule; reported beside report_abcd.py's). Added 2026-10-21
for R25 (registrations/R25_refined_table_agent.md), on the review's finding that the registered interval leaves out
seed-to-seed training variance.

    report_twolevel.py --registration R25star --pairs a:b c:d --out /abs/path/twolevel_R25star.json

The arms are read as report_abcd.py reads them (results/confirm/<registration>/<arm>/seed<k>.npz: pred, y, convo_id),
and the test rows must be equal in every arm. Per resample:
1. each arm's seeds are drawn with replacement (as many as it has);
2. the test dialogues are drawn with replacement;
3. the difference is the mean over the drawn dialogues' turns of (A's correctness averaged over its drawn seeds minus
   B's).
10,000 resamples, numpy default_rng(0). The point estimate is report_abcd.py's (all seeds, all dialogues). Writes --out
and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path):
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
    Z = [np.load(folder / f"seed{k}.npz") for k in seeds]
    y, cid = Z[0]["y"], Z[0]["convo_id"]
    for z in Z[1:]:
        assert np.array_equal(z["y"], y) and np.array_equal(z["convo_id"], cid), folder
    return seeds, np.stack([z["pred"] == y for z in Z]).astype(float), y, cid


def two_level(cA: np.ndarray, cB: np.ndarray, cid: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    uniq, inv = np.unique(cid, return_inverse=True)
    nd = len(uniq)
    SA = np.stack([np.bincount(inv, weights=c, minlength=nd) for c in cA])  # [seeds, dialogues]: correct turns
    SB = np.stack([np.bincount(inv, weights=c, minlength=nd) for c in cB])
    W = np.bincount(inv, minlength=nd).astype(float)
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        ia = rng.integers(0, len(SA), len(SA))
        ib = rng.integers(0, len(SB), len(SB))
        D = SA[ia].mean(0) - SB[ib].mean(0)
        idx = rng.integers(0, nd, nd)
        bs[b] = D[idx].sum() / W[idx].sum()
    point = float((SA.mean(0) - SB.mean(0)).sum() / W.sum())
    return {"diff": point, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; not overwriting")
    base = ROOT / "results" / "confirm" / args.registration
    cache, meta = {}, None
    res = {"what": "two-level bootstrap (seeds within each arm, then dialogues; 10,000; seed 0); no rule", "registration": args.registration, "pairs": {}}
    for p in args.pairs:
        a, b = p.split(":")
        for arm in (a, b):
            if arm not in cache:
                seeds, c, y, cid = load(base / arm)
                if meta is None:
                    meta = (y, cid)
                assert np.array_equal(y, meta[0]) and np.array_equal(cid, meta[1]), arm
                cache[arm] = (seeds, c)
        r = two_level(cache[a][1], cache[b][1], meta[1])
        res["pairs"][p] = {**r, "seeds": [cache[a][0], cache[b][0]]}
        print(f"{p:40s} {100 * r['diff']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] (two-level)", flush=True)
    res["n_test_action_turns"] = int(len(meta[0]))
    res["n_test_conversations"] = int(len(np.unique(meta[1])))
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
