#!/usr/bin/env python
"""W1 engine development (no evaluation data): where the time of one CoTM.feedback minibatch goes.

    profile_step.py --slice S100 --clauses 4000 --batch 256 [--features 20000] [--max-outputs 0]

Builds the same machine and minibatches as mo_tm.py (train slice only), runs 10 warm-up minibatches, then profiles 10
with torch.profiler (CUDA time per operator) and prints the top operators and the total. --max-outputs keeps only
the most frequent output tokens (0 = all). Prints one JSON line; nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_tm as M  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="S100")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--max-outputs", type=int, default=0)
    ap.add_argument("--mem", type=float, default=0.3)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem)
    ids = M.load_slice(args.slice)
    V = M.vocab(ids, 10, args.features)
    if args.max_outputs:
        keep = V["out_ids"][: args.max_outputs]
        V["out_index"][:] = -1
        V["out_index"][keep] = range(len(keep))
        V["out_ids"] = keep
    sub = M.subsample(ids, V["counts"], 1e-4, 0)
    st = M.Stream(sub, V, 5, dev)
    cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=5.0, batch_size=args.batch, neg_sampling="random",
                   negation=False, half=True, seed=1)
    tm = make_tm(len(V["out_ids"]), st.F, cfg, "empty")
    q = M.neg_dist(V, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    perm = st.positions[torch.randperm(st.positions.shape[0], generator=g, device=dev)]

    def step(i):
        X, pos = st.batch(perm[i * args.batch : (i + 1) * args.batch], g)
        neg = torch.multinomial(q, X.shape[0], replacement=True, generator=g)
        tm.feedback(X, pos, None, neg, neg != pos)

    for i in range(10):
        step(i)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(10, 20):
            step(i)
        torch.cuda.synchronize()
    wall = (time.time() - t0) / 10
    rows = []
    for e in prof.key_averages():
        rows.append((e.key, e.device_time_total / 1000 / 10 if hasattr(e, "device_time_total") else e.cuda_time_total / 1000 / 10,
                     e.cpu_time_total / 1000 / 10, e.count / 10))
    rows.sort(key=lambda r: -r[1])
    print(json.dumps({"outputs": len(V["out_ids"]), "features": st.F, "clauses": args.clauses, "batch": args.batch,
                      "wall_ms_per_minibatch": round(1000 * wall, 1),
                      "top_ops_cuda_ms_per_minibatch": [[k, round(c, 2), round(cp, 2), n] for k, c, cp, n in rows[:18]],
                      "stats": tm.stats()}))


if __name__ == "__main__":
    main()
