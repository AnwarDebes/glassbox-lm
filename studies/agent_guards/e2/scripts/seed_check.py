"""E2 training-seed check (rev2, review item 5): a CPU-only reading with NO rule, reported beside every interval.
The registered intervals resample tasks only; they leave out the variance from the TM's training seed (the twins are
deterministic). This refits D4 with seeds 2-5 at each fold's FROZEN configuration (the clauses, s, T and tau stored in
models/final/d4_tm_<fold>.json; same rows, same bits, same epochs) and compares the per-call decisions with seed 1's.
Decision-layer venv, CPU only (2 threads, nice 19):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I -B scripts/seed_check.py \
        --models models/final --rows data/rows_gt_v1.2.2.jsonl runs/E2-1/<stamp>/driver/rows_d0.jsonl \
        --eval-rows runs/E2-1/<stamp>/driver/rows_d0.jsonl --qbits data/query_bits --out models/final/seed_check.json

Per fold (held-out suite h):
- refit with seed 1 and check that it reproduces the stored model exactly (clauses and weights): if not, training is
  not deterministic on this machine and that is reported first;
- for seeds 2-5: the share of h's D0 calls (every executed call of a known tool in h's D0 traces, labelled or not)
  on which the seed's guarded decision (veto clauses + tau, as deployed) equals seed 1's; the open-loop block rate
  (blocked / calls) per seed, over all calls and split by benign and attacked runs; the spread over seeds 1-5.
--epochs is for the code test only (never for the reading).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "scripts"))
import train_loso as TL

from e2guard.bits import QueryBits, vetoes_fired
from e2guard.models import GuardedTM


def eval_rows(paths: list[str], held: str) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["suite"] != held or r["facts"]["kind"] is None:
                continue
            r["_on"] = set(r["facts_on"])
            r["_veto"] = bool(vetoes_fired(r["_on"]))
            r["_y"] = TL.ALLOW if r.get("label") == "ALLOW" else TL.DENY  # unused by the decisions; matrix() needs it
            out.append(r)
    return out


def decisions(model: dict, X: np.ndarray) -> np.ndarray:
    g = GuardedTM(model)
    return np.array([g.decide(x.tolist())["allow"] for x in X], bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--rows", nargs="+", required=True, help="the training rows, exactly as train_loso got them")
    ap.add_argument("--eval-rows", nargs="+", required=True, help="the D0 rows (each fold uses its held-out suite's)")
    ap.add_argument("--qbits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--folds", nargs="*", default=TL.SUITES)
    ap.add_argument("--epochs", type=int, default=0, help="code test only: override the stored epochs")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    t0 = time.time()
    rows = TL.load_rows(args.rows)
    by_suite = {s: [r for r in rows if r["suite"] == s] for s in TL.SUITES}
    res: dict = {"note": "training-seed check; a reading with no rule", "seeds": [1, *args.seeds], "folds": {},
                 "rows_files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.rows},
                 "eval_rows_files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.eval_rows},
                 "test_epochs_override": args.epochs or None}
    for held in args.folds:
        stored = json.loads((Path(args.models) / f"d4_tm_{held}.json").read_text())
        cfg = stored["config"]
        TL.EPOCHS = args.epochs or int(cfg["epochs"])
        qb = QueryBits(Path(args.qbits) / f"fold_{held}.json")
        fit_rows = [r for s in TL.SUITES if s != held for r in by_suite[s] if not r["_veto"]]
        Xf, yf, names = TL.matrix(fit_rows, qb)
        assert names == stored["names"], f"bit names differ from the stored {held} model"
        ev = eval_rows(args.eval_rows, held)
        Xe, _, _ = TL.matrix(ev, qb)
        attacked = np.array([r.get("injection_task") is not None for r in ev], bool)
        fold: dict = {"config": cfg, "fit_rows": len(fit_rows), "eval_calls": len(ev),
                      "eval_calls_attacked": int(attacked.sum()), "epochs_used": TL.EPOCHS}
        base = decisions(stored, Xe)
        refit1 = TL.export_tm(TL.fit_tm(Xf, yf, int(cfg["clauses"]), float(cfg["s"]), 1), names, int(cfg["tau"]))
        fold["seed1_refit_reproduces_stored"] = bool(refit1["clauses"] == stored["clauses"] and refit1["W"] == stored["W"])
        fold["seed1_refit_decisions_equal"] = int((decisions(refit1, Xe) == base).sum())
        per = {1: {"agree_with_seed1": 1.0, "block_rate": float((~base).mean()) if len(base) else None,
                   "block_rate_benign": float((~base[~attacked]).mean()) if (~attacked).any() else None,
                   "block_rate_attacked": float((~base[attacked]).mean()) if attacked.any() else None}}
        for sd in args.seeds:
            m = TL.export_tm(TL.fit_tm(Xf, yf, int(cfg["clauses"]), float(cfg["s"]), sd), names, int(cfg["tau"]))
            d = decisions(m, Xe)
            per[sd] = {"agree_with_seed1": float((d == base).mean()) if len(d) else None,
                       "block_rate": float((~d).mean()) if len(d) else None,
                       "block_rate_benign": float((~d[~attacked]).mean()) if (~attacked).any() else None,
                       "block_rate_attacked": float((~d[attacked]).mean()) if attacked.any() else None}
            print(f"[{held}] seed {sd}: {json.dumps(per[sd])} ({time.time() - t0:.0f}s)", flush=True)
        br = [v["block_rate"] for v in per.values() if v["block_rate"] is not None]
        fold["per_seed"] = per
        fold["block_rate_spread"] = {"min": min(br), "max": max(br), "sd": statistics.pstdev(br)} if br else None
        fold["min_agreement"] = min(v["agree_with_seed1"] for k, v in per.items() if k != 1) if args.seeds else None
        res["folds"][held] = fold
        out.write_text(json.dumps(res, indent=1, default=str))
    res["minutes"] = round((time.time() - t0) / 60, 2)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: {"reproduces": v["seed1_refit_reproduces_stored"], "min_agreement": v["min_agreement"],
                          "block_rate_spread": v["block_rate_spread"]} for k, v in res["folds"].items()}, indent=1))


if __name__ == "__main__":
    main()
