#!/usr/bin/env python
"""W1 diagnostic (no bar, no evaluation data): what a word table knows, measured on held-out C4 text only.

    probe_table.py --table tables/NAME.npz [--table potion] --slice S10

1. Held-out ranking with the table's vectors alone (comparable across the TM, the twin, SGNS and potion): for 10,000
   positions of the held-out slice H (the subsampling rule and the output list of --slice, full window), the
   score of a candidate token c is cos(t_c, mean of t_x over the context tokens x that have a vector); the true token
   is ranked against 99 unigram^0.75 negatives (the candidates of mo_tm.heldout_set, same seeds). Mean reciprocal
   rank and strict top-1 share, over the positions whose true token and at least one context token have a vector.
2. Nearest neighbours (cosine) of test 5's probe words: balance transfer alarm weather french card music flight
   cancel email.
One row per table in results_w1_probe.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_tm as M  # noqa: E402

PROBES = ["balance", "transfer", "alarm", "weather", "french", "card", "music", "flight", "cancel", "email"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="append", required=True)
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100"))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--heldout", type=int, default=10000)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.2)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots")
                                  / "1e5a03f8eeb2c98b928fbbd846f22f816360919f" / "tokenizer.json"))
    ids = M.load_slice(args.slice)
    V = M.vocab(ids, args.min_count, args.features)
    del ids
    h = M.subsample(M.load_slice("H"), V["counts"], args.sub_t, args.sub_seed + 1)
    st = M.Stream(h, V, args.window, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    pick = st.positions[torch.randperm(st.positions.shape[0], generator=g, device=dev)[: args.heldout]]
    Q = (pick.unsqueeze(1) + st.off.unsqueeze(0)).clamp(0, st.ids.shape[0] - 1)
    ok = (st.ids[Q] >= 0) & (st.doc[Q] == st.doc[pick].unsqueeze(1)) & (st.ids[Q] != st.ids[pick].unsqueeze(1))
    ctx_tok = torch.where(ok, st.ids[Q], torch.full_like(Q, -1))
    y_tok = st.ids[pick]
    q = M.neg_dist(V, dev)
    g.manual_seed(11)
    yi = st.y[pick]
    cand = torch.multinomial(q, pick.shape[0] * 99, replacement=True, generator=g).view(-1, 99)
    while bool((cand == yi.unsqueeze(1)).any()):
        bad = cand == yi.unsqueeze(1)
        cand[bad] = torch.multinomial(q, int(bad.sum()), replacement=True, generator=g)
    cand_tok = torch.as_tensor(V["out_ids"], device=dev)[cand]
    for spec in args.table:
        if spec == "potion":
            from sentence_transformers import SentenceTransformer

            stm = SentenceTransformer("minishlab/potion-base-32M", device="cpu", revision="1e5a03f8eeb2c98b928fbbd846f22f816360919f")
            full = stm[0].embedding.weight.detach().float().numpy()
            vocab_ids, codes, name = np.arange(full.shape[0]), full, "potion-base-32M"
        else:
            z = np.load(HERE / spec)
            vocab_ids, codes, name = z["vocab_ids"].astype(np.int64), z["codes"].astype(np.float32), Path(spec).stem
        T = torch.zeros((M.V_POTION, codes.shape[1]), device=dev)
        T[torch.as_tensor(vocab_ids, device=dev)] = torch.as_tensor(codes, device=dev)
        T = torch.nn.functional.normalize(T, dim=1)
        has = T.abs().sum(1) > 0
        gts, eqs, us = [], [], []
        for a in range(0, ctx_tok.shape[0], 250):  # chunks: a 4,000-dimensional table does not fit in one piece
            ct = ctx_tok[a : a + 250]
            w = (has[ct.clamp(min=0)] & (ct >= 0)).float().unsqueeze(2)
            m = torch.nn.functional.normalize((T[ct.clamp(min=0)] * w).sum(1), dim=1)
            us.append(has[y_tok[a : a + 250]] & (m.abs().sum(1) > 0))
            s0 = (T[y_tok[a : a + 250]] * m).sum(1, keepdim=True)
            sc = torch.einsum("bkd,bd->bk", T[cand_tok[a : a + 250]], m)
            gts.append((sc > s0).sum(1).float())
            eqs.append((sc == s0).sum(1).float())
        gt, eq, usable = torch.cat(gts), torch.cat(eqs), torch.cat(us)
        rr = (1.0 / (1.0 + gt + 0.5 * eq))[usable]
        top1 = ((gt == 0) & (eq == 0)).float()[usable]
        neigh = {}
        for w in PROBES:
            t = tok.token_to_id(w)
            if t is None or not bool(has[t]):
                neigh[w] = None
                continue
            s = T @ T[t]
            s[t] = -2
            top = s.topk(6).indices.cpu().numpy()
            neigh[w] = [tok.id_to_token(int(i)) for i in top]
        rec = {"table": name, "slice_for_vocab": args.slice, "heldout_positions": int(pick.shape[0]),
               "usable_positions": int(usable.sum()), "mrr": round(float(rr.mean()), 4), "top1": round(float(top1.mean()), 4),
               "rows_with_vector": int(has.sum()), "neighbours": neigh, "utc": M.utc()}
        print(json.dumps(rec), flush=True)
        with open(HERE / "results_w1_probe.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
