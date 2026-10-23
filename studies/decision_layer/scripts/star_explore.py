#!/usr/bin/env python
"""TM settings on the cached readable bits (src/dl/bitcache.py), scored on DEV only.

Like scripts/abcd_explore.py's ``tm`` arm (same view, same TMConfig fields, best-dev snapshot),
plus ``--balance RHO`` (class-balanced draws, src/dl/tm_balance.py), and the dev accuracy split
by how common the true action is in train. The test split is not read. One JSON line per run to
results/explore/abcd.jsonl (field "script": "star_explore.py"). Added 2026-10-17 after R18.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.bitcache import view_bits  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_balance import fit_balanced  # noqa: E402
from dl.tm_init import INITS, make_tm  # noqa: E402

BINS = [(1, 20), (21, 100), (101, 400), (401, 1500), (1501, 10**9)]


def cap_gpu() -> None:
    """Cap this process's share of the shared GPU (env DL_GPU_FRAC, default 0.15 of the card), so one job
    cannot starve other runs of memory."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DL_GPU_FRAC", "0.15")))


def main() -> None:
    cap_gpu()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="star", choices=["abcd", "star"])
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--init", default="half", choices=INITS)
    ap.add_argument("--balance", type=float, default=-1.0, help="rho >= 0: class-balanced draws; < 0: CoTM.fit (a permutation per epoch)")
    ap.add_argument("--neg", default="hardest", help="negative-class rule (TMConfig.neg_sampling)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-chunk", action="store_true", help="unchunked automaton update above dl.tm.CHUNK_ELEMS (needed for a literal "
                    "budget with many clauses; set for this process only, tm.py unchanged; more GPU memory)")
    ap.add_argument("--chunk-budget", action="store_true", help="the exact literal budget inside tm.py's clause-chunked update "
                    "(src/dl/tm_chunkbudget.py); needed above CHUNK_ELEMS clause-literal pairs")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.no_chunk:
        import dl.tm as tm_mod

        tm_mod.CHUNK_ELEMS = 2**62
    parts = tuple(p for p in args.parts.split(",") if p)
    Xtr, ytr, Xdv, ydv, names, blist, tr, dv = view_bits(args.dataset, parts)
    C_ = len(blist)
    cfg = TMConfig(n_clauses=args.clauses, T=args.T or args.clauses, s=args.s, epochs=args.epochs, max_literals=args.max_literals,
                   budget_select="rank", neg_sampling=args.neg, seed=args.seed)
    tm = make_tm(C_, Xtr.shape[1], cfg, init=args.init)
    if args.chunk_budget:
        from dl.tm_chunkbudget import enable

        enable(tm)
    if args.balance >= 0:
        fit_balanced(tm, Xtr, ytr, Xdv, ydv, rho=args.balance)
    else:
        tm.fit(Xtr, ytr, Xdv, ydv)
    pred = tm.predict(torch.as_tensor(Xdv, dtype=torch.uint8, device="cuda")).cpu().numpy()
    ok = pred == ydv
    cnt = np.bincount(ytr, minlength=C_)
    by = {f"{lo}-{hi}": float(ok[(cnt[ydv] >= lo) & (cnt[ydv] <= hi)].mean()) for lo, hi in BINS if ((cnt[ydv] >= lo) & (cnt[ydv] <= hi)).any()}
    rec = {"tag": args.tag, "script": "star_explore.py", "dataset": args.dataset, "parts": parts, "seed": args.seed, "arm": "tm",
           "acc": float(ok.mean()), "balance": args.balance, "by_train_freq": by, "cfg": asdict(cfg), "init": args.init,
           "best_epoch": getattr(tm, "best_epoch", None), "history": tm.history, "train_s": round(tm.train_seconds, 1),
           "stats": tm.stats(), "n_bits": int(Xtr.shape[1]), "no_chunk": args.no_chunk, "chunk_budget": args.chunk_budget, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (ROOT / "results" / "explore" / "abcd.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("history", "cfg")}), flush=True)


if __name__ == "__main__":
    main()
