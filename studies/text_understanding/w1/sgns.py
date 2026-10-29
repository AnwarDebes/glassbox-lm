#!/usr/bin/env python
"""W1 reference (no bar): word2vec skip-gram with negative sampling (SGNS, a shallow neural network) on the same
slice, the same subsampled stream and the same output vocabulary as the Tsetlin machine (mo_tm.py), so that the
curve shows what a neural predictor gains from the same data. No SGNS number enters a claim about the TM.

    sgns.py --slice S10 --tag NAME [--dim 300 --neg 5 --epochs 5 --lr 0.025]

Pairs: for each position whose token is an output token, every other output token within a dynamic window
(b uniform in 1..W on each side, inside the document; the same type is skipped, as the TM masks it), both as
(center, context). Loss: -log sigmoid(u_c . v_o) - sum over neg draws from unigram^0.75 of log sigmoid(-u_c . v_n).
Plain SGD on sparse rows, learning rate falling linearly from --lr to --lr * 1e-4 over the run (word2vec's schedule),
minibatches of --pairs pairs, torch generator seeded by --seed. The table = the input vectors u, L2-normalized.
Written as tables/{tag}.npz (mo_tm.py's format); one row in results_w1_train.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_tm as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--pairs", type=int, default=65536)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.15)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem)
    ids = M.load_slice(args.slice)
    V = M.vocab(ids, args.min_count, args.features)
    sub = M.subsample(ids, V["counts"], args.sub_t, args.sub_seed)
    st = M.Stream(sub, V, args.window, dev)
    n_out = len(V["out_ids"])
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    q = M.neg_dist(V, dev)
    U = ((torch.rand((n_out, args.dim), generator=g, device=dev) - 0.5) / args.dim).requires_grad_(False)
    Vc = torch.zeros((n_out, args.dim), device=dev)
    n = st.ids.shape[0]
    P = st.positions
    total_steps, step = None, 0
    t1 = time.time()
    for ep in range(args.epochs):
        perm = P[torch.randperm(P.shape[0], generator=g, device=dev)]
        per = max(1, args.pairs // (2 * args.window))  # positions per minibatch
        if total_steps is None:
            total_steps = args.epochs * ((P.shape[0] + per - 1) // per)
        for a in range(0, perm.shape[0], per):
            c = perm[a : a + per]
            Q = c.unsqueeze(1) + st.off.unsqueeze(0)
            inside = (Q >= 0) & (Q < n)
            Qc = Q.clamp(0, n - 1)
            b = torch.randint(1, args.window + 1, (c.shape[0], 1), generator=g, device=dev)
            ok = inside & (st.doc[Qc] == st.doc[c].unsqueeze(1)) & (st.dist.unsqueeze(0) <= b)
            yo = st.y[Qc]
            ok &= (yo >= 0) & (st.ids[Qc] != st.ids[c].unsqueeze(1))
            ci = st.y[c].unsqueeze(1).expand_as(yo)[ok]
            oi = yo[ok]
            lr = args.lr * max(1e-4, 1.0 - step / total_steps)
            u = U[ci]
            vo = Vc[oi]
            negs = torch.multinomial(q, ci.shape[0] * args.neg, replacement=True, generator=g).view(-1, args.neg)
            vn = Vc[negs]
            gpos = torch.sigmoid((u * vo).sum(1)) - 1.0  # d loss / d score, positive pair
            gneg = torch.sigmoid((u.unsqueeze(1) * vn).sum(2))  # negatives
            du = gpos.unsqueeze(1) * vo + (gneg.unsqueeze(2) * vn).sum(1)
            dvo = gpos.unsqueeze(1) * u
            dvn = gneg.unsqueeze(2) * u.unsqueeze(1)
            U.index_add_(0, ci, -lr * du)
            Vc.index_add_(0, oi, -lr * dvo)
            Vc.index_add_(0, negs.reshape(-1), -lr * dvn.reshape(-1, args.dim))
            step += 1
    torch.cuda.synchronize()
    train_s = time.time() - t1
    (HERE / "tables").mkdir(exist_ok=True)
    p = HERE / "tables" / f"{args.tag}.npz"
    assert not p.exists(), f"{p} exists: a new version needs a new tag"
    codes = torch.nn.functional.normalize(U, dim=1).cpu().numpy().astype(np.float16)
    np.savez(p, vocab_ids=V["out_ids"].astype(np.int32), codes=codes, counts=V["counts"][V["out_ids"]].astype(np.int64))
    rec = {"tag": args.tag, "kind": "SGNS reference (neural, no bar)", "slice": args.slice, "utc": M.utc(),
           "cfg": vars(args), "outputs": n_out, "tokens_after_subsampling": int((sub >= 0).sum()), "steps": step,
           "train_s": round(train_s, 1), "total_s": round(time.time() - t0, 1), "table": str(p.relative_to(HERE)),
           "table_sha256": M.sha16(p), "sha256": {"sgns.py": M.sha16(Path(__file__).resolve()), "mo_tm.py": M.sha16(HERE / "mo_tm.py")}}
    print(json.dumps(rec), flush=True)
    with open(HERE / "results_w1_train.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
