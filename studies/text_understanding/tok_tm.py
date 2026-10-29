#!/usr/bin/env python
"""Test 3 (LOG.md): token clauses, the Graph Tsetlin Machine's reading of a sentence, batched on the study's CoTM.

    tok_tm.py --dataset clinc150 --parts S|G+S|S+P+N|G+S+P+N|W+P+N... --seed 1 --emb-dir DIR --tag TAG [--test] [--smoke]

A clause holds at a token; a sentence matches the clause if some token does (existential over the words, as a
Graph TM clause over word nodes, arXiv 2507.14874 Sec. 3.1; with P and N it also sees the neighbours' symbols,
as the convolutional TM sees a window). Per token the TM sees the parts named in --parts:
- S: the token's meaning bits (graph_feats.py: "cos(token, concept k) >= t", the thresholds of existsT);
- P / N: the S bits of the previous / next token (all 0 at the sentence edge);
- G: the existsT bits of the whole sentence (the OR of S over its tokens; the same for every token);
- W: the token itself (one bit per training token; tokens unseen in training have none);
- V: the sentence's bag of training tokens (the OR of W over its tokens; the same for every token), so that a
     clause can also join several words, as a flat bag-of-words TM can (W alone holds one word per clause).
Why not the stock GraphTsetlinMachine: it trains one sentence at a time with about 10 synchronised kernel
launches each; on the shared GPU a launch plus synchronise took 12 ms (median, launch_probe.py), so 200 ms per
sentence (prof_graph.py). This file keeps the CoTM's batch rule (dl/tm.py, imported read only) and changes only
where a clause is evaluated and which input its feedback sees: the literals of one random token at which the
fired clause holds (the Graph TM's rule, select_clause_node). Settings as test 2 (4,000 clauses, T 4000, s 8,
40 epochs, batch 128, hardest negative, negation, init half, eval every 2). The epoch is chosen on val and the
state of that epoch restored; --test scores the test split once. Appends one line to results_tok.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import os
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on the shared GPU
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import load  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

# The GPU is shared and was full (0.03 GiB free) at 16:25 UTC: the automata update runs in smaller clause chunks
# than dl/tm.py (same rule and distributions, another random stream) and the process is capped, so that this job
# fails before it can push another job out of memory.
TOK_CHUNK_ELEMS = 10_000_000
ROWS = 4096  # token rows per clause-evaluation product
MEM_FRACTION = 0.1


class Feats:
    """Per-token input bits built on the GPU from a token table, one batch at a time."""

    def __init__(self, z, parts: str, splits, dev):
        self.parts = parts.split("+")
        self.dev = dev
        n_sym = int(z["n_sym"])
        used = z["used"]
        U = len(used)
        tab = np.unpackbits(z["node_bits"], axis=1)[:, :n_sym]
        self.tab = torch.as_tensor(np.vstack([tab, np.zeros((1, n_sym), np.uint8)]), device=dev)  # row U: none
        lut = np.full(int(used.max()) + 1, U, np.int64)
        lut[used] = np.arange(U)
        wv = z["train_vocab"]
        wl = np.full(int(used.max()) + 1, -1, np.int64)
        wl[wv] = np.arange(len(wv))
        self.n_sym, self.n_w, self.pad = n_sym, len(wv), U
        self.tok, self.wid, self.lens, self.G, self.V = {}, {}, {}, {}, {}
        for sp in splits:
            tok, off = z[f"{sp}__tok"], z[f"{sp}__off"]
            lens = np.diff(off)
            Lm = max(1, int(lens.max()))
            T = np.full((len(lens), Lm), U, np.int64)
            Wd = np.full((len(lens), Lm), -1, np.int64)
            for i in range(len(lens)):
                t = tok[off[i] : off[i + 1]]
                T[i, : len(t)] = lut[t]
                Wd[i, : len(t)] = wl[t]
            self.tok[sp] = torch.as_tensor(T, device=dev)
            self.wid[sp] = torch.as_tensor(Wd, device=dev)
            self.lens[sp] = torch.as_tensor(lens, device=dev)
            if "G" in self.parts:  # OR over the tokens = the existsT vector (graph_feats.py checks the identity)
                G = torch.zeros((len(lens), n_sym), dtype=torch.uint8, device=dev)
                for a in range(0, len(lens), 1024):
                    G[a : a + 1024] = self.tab[self.tok[sp][a : a + 1024]].amax(1)
                self.G[sp] = G
            if "V" in self.parts:  # OR over the tokens of the word bits = the sentence's bag of training words
                V = torch.zeros((len(lens), len(wv) + 1), dtype=torch.uint8, device=dev)
                Wt = self.wid[sp]
                V.scatter_(1, torch.where(Wt >= 0, Wt, len(wv)), 1)
                self.V[sp] = V[:, : len(wv)].contiguous()
        width = {"S": n_sym, "P": n_sym, "N": n_sym, "G": n_sym, "W": len(wv), "V": len(wv)}
        self.f = sum(width[p] for p in self.parts)
        self.names = []
        for p in self.parts:
            self.names += [f"{p}{j}" for j in range(width[p])]

    def batch(self, sp, idx):
        """uint8 bits [b, L, f] and the valid-token mask [b, L] for the sentences idx of split sp."""
        lens = self.lens[sp][idx]
        L = max(1, int(lens.max()))
        T = self.tok[sp][idx, :L]
        b = T.shape[0]
        mask = torch.arange(L, device=self.dev).unsqueeze(0) < lens.unsqueeze(1)
        mask[:, 0] |= lens == 0  # an empty sentence: one token without symbols
        out = []
        for p in self.parts:
            if p == "S":
                out.append(self.tab[T])
            elif p == "P":
                Tp = torch.cat([torch.full((b, 1), self.pad, device=self.dev, dtype=T.dtype), T[:, :-1]], 1)
                out.append(self.tab[Tp])
            elif p == "N":
                Tn = torch.cat([T[:, 1:], torch.full((b, 1), self.pad, device=self.dev, dtype=T.dtype)], 1)
                out.append(self.tab[Tn])
            elif p == "G":
                out.append(self.G[sp][idx].unsqueeze(1).expand(b, L, self.n_sym))
            elif p == "V":
                out.append(self.V[sp][idx].unsqueeze(1).expand(b, L, self.n_w))
            elif p == "W":
                Wd = self.wid[sp][idx, :L]
                oh = torch.zeros((b, L, self.n_w + 1), dtype=torch.uint8, device=self.dev)
                oh.scatter_(2, torch.where(Wd >= 0, Wd, self.n_w).unsqueeze(2), 1)  # column n_w: unseen tokens, dropped
                out.append(oh[:, :, : self.n_w])
        return torch.cat(out, 2), mask


class TokTM(CoTM):
    """The CoTM of dl/tm.py (state, weights, update rule) with clauses evaluated per token."""

    def tok_outputs(self, X, mask, train_rule, rows: int = ROWS):
        """Clause outputs per token [b, L, N] (bool), computed over at most ``rows`` token rows at a time."""
        b, L, f = X.shape
        X2 = X.reshape(b * L, f)
        incl = self._incl()
        out = torch.empty((b * L, self.cfg.n_clauses), dtype=torch.bool, device=self.dev)
        for a in range(0, b * L, rows):
            Lt = self._literals(X2[a : a + rows])
            out[a : a + rows] = ((1 - Lt) @ incl.T) == 0
            del Lt
        co = out.reshape(b, L, -1) & mask.unsqueeze(2)
        if not train_rule:
            co = co & (incl.sum(1) > 0).view(1, 1, -1)
        return co

    def scores(self, F: Feats, sp: str):
        out = []
        wf = self.w.float()
        n = len(F.lens[sp])
        bs = max(8, ROWS // max(1, int(F.lens[sp].max())))
        for a in range(0, n, bs):
            idx = torch.arange(a, min(n, a + bs), device=self.dev)
            X, mask = F.batch(sp, idx)
            co = self.tok_outputs(X, mask, False)
            out.append(co.any(1).float() @ wf)
        return torch.cat(out)

    def step(self, F: Feats, idx, yb):
        """The supervised rule of CoTM.feedback (no budgets, no clause drop), clauses over tokens."""
        cfg = self.cfg
        T, n_states, N = cfg.T, cfg.n_states, cfg.n_clauses
        X, mask = F.batch("train", idx)
        b, Lb, f = X.shape
        co_tok = self.tok_outputs(X, mask, True)
        co = co_tok.any(1)  # [b, N]
        S = co.float() @ self.w.float()
        yn = self._neg_classes(S, yb)
        ar = torch.arange(b, device=self.dev)
        Sp = S[ar, yb].clamp(-T, T)
        Sn = S[ar, yn].clamp(-T, T)
        p_pos = (T - Sp) / (2.0 * T)
        p_neg = (T + Sn) / (2.0 * T)
        Wt = self.w[:, yb].T
        Wn = self.w[:, yn].T
        sel_pos = torch.rand((b, N), generator=self.gen, device=self.dev) < p_pos.unsqueeze(1)
        sel_neg = torch.rand((b, N), generator=self.gen, device=self.dev) < p_neg.unsqueeze(1)
        t1 = (sel_pos & (Wt >= 0)) | (sel_neg & (Wn < 0))
        g1 = t1 & co
        g1nf = t1 & ~co
        g2 = ((sel_pos & (Wt < 0)) | (sel_neg & (Wn > 0))) & co
        inc_t = (sel_pos & co).float()
        inc_n = (sel_neg & co).float()
        dW = torch.zeros((N, self.C), device=self.dev)
        dW.index_add_(1, yb, inc_t.T)
        dW.index_add_(1, yn, -inc_n.T)
        self.w += dW.to(torch.int32)
        # one random token at which each fired clause holds: its literals are the input of the feedback
        keys = torch.rand((b, Lb, N), generator=self.gen, device=self.dev).masked_fill_(~co_tok, -1.0)
        pick = keys.argmax(1)  # [b, N]
        del keys
        sel = torch.zeros((b, Lb, N), dtype=torch.bool, device=self.dev)
        sel.scatter_(1, pick.unsqueeze(1), True)
        dt = self.mm_dtype
        A1 = (sel & g1.unsqueeze(1)).reshape(b * Lb, N).to(dt)
        A2 = (sel & g2.unsqueeze(1)).reshape(b * Lb, N).to(dt)
        del sel
        # The literals are [x, 1 - x], so A^T @ literals = [A^T x, c - A^T x] and A^T @ (1 - literals) =
        # [c - A^T x, A^T x] with c = the column sums of A: one product with the plain bits, exact (every
        # entry is a count of at most b, and fp16 holds integers up to 2048 exactly).
        Xf = X.reshape(b * Lb, f).to(dt)
        c1, c2 = A1.float().sum(0).unsqueeze(1), A2.float().sum(0).unsqueeze(1)
        n1nf = g1nf.float().sum(0)
        cs = max(1, TOK_CHUNK_ELEMS // self.m)
        for a in range(0, N, cs):
            e = min(N, a + cs)
            ta = self.ta[a:e]
            AX1 = (A1[:, a:e].T @ Xf).float()
            cnt_inc = torch.cat([AX1, c1[a:e] - AX1], 1)
            cnt_dec = torch.cat([c1[a:e] - AX1, AX1], 1) + n1nf[a:e].unsqueeze(1)
            del AX1
            AX2 = (A2[:, a:e].T @ Xf).float()
            cnt_t2 = torch.cat([c2[a:e] - AX2, AX2], 1)
            del AX2
            excluded = ta <= n_states
            inc = torch.binomial(cnt_inc, torch.full_like(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
            del cnt_inc
            dec = torch.binomial(cnt_dec, torch.full_like(cnt_dec, 1.0 / cfg.s), generator=self.gen)
            del cnt_dec
            self.ta[a:e] = (ta + (inc - dec + cnt_t2 * excluded.float()).to(torch.int32)).clamp_(1, 2 * n_states)

    def fit_tok(self, F: Feats, y_tr, y_va, epochs, patience: int = 0):
        """patience > 0: stop after that many evaluations in a row without a new best val (0: run all epochs)."""
        cfg = self.cfg
        t0 = time.time()
        yt = torch.as_tensor(y_tr, dtype=torch.long, device=self.dev)
        yv = torch.as_tensor(y_va, dtype=torch.long, device=self.dev)
        n = len(yt)
        best, state, stale = -1.0, None, 0
        self.stopped_epoch = epochs
        for ep in range(epochs):
            perm = torch.randperm(n, generator=self.gen, device=self.dev)
            for a in range(0, n, cfg.batch_size):
                idx = perm[a : a + cfg.batch_size]
                self.step(F, idx, yt[idx])
            if ep % cfg.eval_every == cfg.eval_every - 1 or ep == epochs - 1:
                acc = float((self.scores(F, "val").argmax(1) == yv).float().mean())
                self.history.append({"epoch": ep + 1, "val_acc": round(acc, 4), "t": round(time.time() - t0, 1)})
                print(f"epoch {ep + 1} val {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if acc > best:
                    best, state, stale = acc, (self.ta.clone(), self.w.clone(), ep + 1), 0
                else:
                    stale += 1
                if patience and stale >= patience:
                    self.stopped_epoch = ep + 1
                    break
        self.ta, self.w, self.best_epoch = state
        self.best_val = best
        torch.cuda.synchronize()
        self.train_seconds = time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40, help="the most epochs")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without a new best val (0: never)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on the first 2,000 training sentences, nothing written")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    t0 = time.time()
    dev = "cuda"
    z = np.load(args.emb_dir / f"{args.dataset}__graph_K{args.K}.npz")
    ds = load(args.dataset)
    splits = ("train", "val", "test") if args.test else ("train", "val")
    for sp in splits:
        assert (z[f"y_{sp}"] == ds.labels[sp]).all()
    F = Feats(z, args.parts, splits, dev)
    y_tr, y_va = z["y_train"], z["y_val"]
    if args.smoke:
        keep = 2000
        F.tok["train"], F.lens["train"], F.wid["train"] = F.tok["train"][:keep], F.lens["train"][:keep], F.wid["train"][:keep]
        if "train" in F.G:
            F.G["train"] = F.G["train"][:keep]
        y_tr = y_tr[:keep]
    cfg = TMConfig(n_clauses=args.clauses, T=4000, s=8.0, epochs=args.epochs, batch_size=128, neg_sampling="hardest",
                   negation=True, max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0,
                   clause_drop=0.0, half=True, seed=args.seed, eval_every=2)
    tm = TokTM(len(ds.intents), F.f, cfg)
    tm.fit_tok(F, y_tr, y_va, 2 if args.smoke else args.epochs, args.patience)
    yv = torch.as_tensor(y_va, device=dev)
    check = float((tm.scores(F, "val").argmax(1) == yv).float().mean())
    rec = {"tag": args.tag, "dataset": args.dataset, "parts": args.parts, "seed": args.seed, "n_features": F.f,
           "cfg": asdict(cfg), "val_acc": round(tm.best_val, 4), "val_acc_check": round(check, 4),
           "best_epoch": tm.best_epoch, "patience": args.patience, "stopped_epoch": tm.stopped_epoch,
           "curve": [[h["epoch"], h["val_acc"]] for h in tm.history]}
    if args.test:
        yt = torch.as_tensor(z["y_test"], device=dev)
        rec["test_acc"] = round(float((tm.scores(F, "test").argmax(1) == yt).float().mean()), 4)
        rec["n_test"] = int(len(yt))
    rec["train_s"], rec["total_s"] = round(tm.train_seconds, 1), round(time.time() - t0, 1)
    if args.smoke:
        print(json.dumps({"smoke": True, **{k: rec[k] for k in ("parts", "n_features", "val_acc", "val_acc_check", "train_s")}}))
        return
    with open(HERE / "results_tok.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve")}))


if __name__ == "__main__":
    main()
