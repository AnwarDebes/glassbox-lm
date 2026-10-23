#!/usr/bin/env python
"""LLM routing on RouterBench (0-shot): choose, per prompt, a cheap LLM or GPT-4.

    rb_run.py --router tm|lr|knn --seed 1 [--cheap zero-one-ai/Yi-34B-Chat] [--view thermo:8|lex:2000|...]

Label for the full-information routers: 1 if GPT-4's score is strictly higher than the cheap
model's score on that prompt, else 0. Each router outputs a score s(x) (higher = send to GPT-4).
Sweeping a threshold over s gives the quality-cost curve on the eval split:
  quality(f) = mean score when the fraction f of prompts with the highest s go to GPT-4.
Reported: area under quality(f) for f in [0, 1] (trapezoid, 101 points), quality at f = 0.2 /
0.3 / 0.5, and the same for random routing, the oracle and the per-task best routing that knows the
task name (an upper reference that routers do not get).
Split: random 60/10/30 (seed 0) of the 36,497 prompts. The task name (eval_name) is never an input.
Prompt embeddings (all-MiniLM-L6-v2, first 256 tokens) are cached in data/routerbench/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

DEV = "cuda"
STRONG = "gpt-4-1106-preview"


def load(enc: str = "minilm"):
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "routerbench" / "routerbench_0shot.parquet")
    p = ROOT / "data" / "routerbench" / f"emb_{enc}.npy"
    if p.exists():
        E = np.load(p)
    else:
        from sentence_transformers import SentenceTransformer

        hf = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge-base": "BAAI/bge-base-en-v1.5"}[enc]
        m = SentenceTransformer(hf, device=DEV)
        E = m.encode(df["prompt"].tolist(), batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
        np.save(p, E)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    n = len(df)
    split = np.empty(n, dtype=object)
    split[perm[: int(0.6 * n)]] = "train"
    split[perm[int(0.6 * n): int(0.7 * n)]] = "val"
    split[perm[int(0.7 * n):]] = "test"
    return df, E, split


def curve(s: np.ndarray, q_cheap: np.ndarray, q_strong: np.ndarray, rng=None):
    """quality(f) for f = 0..1 (101 points): top-f by s go to the strong model."""
    n = len(s)
    order = np.argsort(-s, kind="stable") if rng is None else rng.permutation(n)
    gain = (q_strong - q_cheap)[order]
    cum = np.r_[0, np.cumsum(gain)]
    fs = np.linspace(0, 1, 101)
    k = np.round(fs * n).astype(int)
    return fs, (q_cheap.sum() + cum[k]) / n


def summarise(s, qc, qs):
    fs, qf = curve(s, qc, qs)
    return {"auc": float(np.trapezoid(qf, fs)), "q20": float(qf[20]), "q30": float(qf[30]), "q50": float(qf[50]),
            "q0": float(qf[0]), "q100": float(qf[100])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--cheap", default="zero-one-ai/Yi-34B-Chat")
    ap.add_argument("--enc", default="minilm")
    ap.add_argument("--view", default="thermo:8")
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", default="val")
    ap.add_argument("--out", default="explore")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.eval == "test" and args.out == "explore":
        raise SystemExit("test split only in registered runs")
    t0 = time.time()
    df, E, split = load(args.enc)
    qc = df[args.cheap].to_numpy(float)
    qs = df[STRONG].to_numpy(float)
    y = (qs > qc).astype(np.int64)
    m = {k: split == k for k in ("train", "val", "test")}
    ev = args.eval
    rec = {"tag": args.tag, "router": args.router, "cheap": args.cheap, "enc": args.enc, "seed": args.seed, "eval": ev,
           "n": {k: int(v.sum()) for k, v in m.items()}, "pos_rate_train": float(y[m["train"]].mean())}
    # references on the eval split
    ref = {"cheap_only": float(qc[m[ev]].mean()), "strong_only": float(qs[m[ev]].mean()),
           "oracle": summarise((qs - qc)[m[ev]], qc[m[ev]], qs[m[ev]])}
    rng = np.random.default_rng(args.seed)
    fs, qr = curve(np.zeros(m[ev].sum()), qc[m[ev]], qs[m[ev]], rng=rng)
    ref["random"] = {"auc": float(np.trapezoid(qr, fs)), "q20": float(qr[20]), "q30": float(qr[30]), "q50": float(qr[50])}
    # per-task reference: tasks ranked by mean train gain of the strong model (uses the task name)
    tasks = df["eval_name"].to_numpy()
    gain_tr = {t: float((qs - qc)[m["train"] & (tasks == t)].mean()) if (m["train"] & (tasks == t)).any() else 0.0 for t in np.unique(tasks)}
    s_task = np.array([gain_tr[t] for t in tasks[m[ev]]]) + 1e-6 * rng.random(m[ev].sum())
    ref["per_task_known_name"] = summarise(s_task, qc[m[ev]], qs[m[ev]])
    rec["ref"] = ref

    if args.router == "lr":
        best = (-1, None)
        for Cc in [0.1, 1.0, 10.0, 100.0]:
            W, b = fit_logreg(E[m["train"]], y[m["train"]], 2, Cc)
            s = torch.softmax(logits(E[m["val"]], W, b), 1)[:, 1].cpu().numpy()
            auc = summarise(s, qc[m["val"]], qs[m["val"]])["auc"]
            if auc > best[0]:
                best = (auc, (Cc, W, b))
        Cc, W, b = best[1]
        s = torch.softmax(logits(E[m[ev]], W, b), 1)[:, 1].cpu().numpy()
        rec["C"] = Cc
    elif args.router == "knn":
        A = torch.as_tensor(E[m["train"]], device=DEV)
        yy = torch.as_tensor(y[m["train"]], device=DEV).float()
        Bq = torch.as_tensor(E[m[ev]], device=DEV)
        out = []
        for i in range(0, Bq.shape[0], 2048):
            top = (Bq[i:i + 2048] @ A.T).topk(args.k, dim=1).indices
            out.append(yy[top].mean(1))
        s = torch.cat(out).cpu().numpy()
        rec["k"] = args.k
    elif args.router == "tm":
        texts = df["prompt"].tolist()
        view = make_view(args.view, seed=args.seed)
        tr_idx = np.flatnonzero(m["train"])
        view.fit(E[m["train"]], y[m["train"]], texts=[texts[i] for i in tr_idx], class_names=["cheap", "gpt4"])
        def enc(mask):
            ids = np.flatnonzero(mask)
            return view.transform(E[mask], [texts[i] for i in ids])
        Btr, Bva, Bev = enc(m["train"]), enc(m["val"]), enc(m[ev])
        cfg = TMConfig(n_clauses=args.clauses, T=args.T or args.clauses, s=args.s, epochs=args.epochs,
                       max_literals=args.max_literals, seed=args.seed, eval_every=2)
        tm = CoTM(2, Btr.shape[1], cfg).fit(Btr, y[m["train"]], Bva, y[m["val"]])
        S = tm.class_sums(torch.as_tensor(Bev, dtype=torch.uint8, device=DEV)).cpu().numpy()
        s = S[:, 1] - S[:, 0]
        rec.update({"view": args.view, "clauses": args.clauses, "s": args.s, "epochs": args.epochs,
                    "max_literals": args.max_literals, "val_acc": tm.best_val, "stats": tm.stats()})
        np.savez_compressed(ROOT / "results" / args.out / f"rb_tm_{args.tag or args.seed}.npz", **tm.state(),
                            names=np.array(view.names))
    else:
        raise SystemExit(args.router)
    s = s + 1e-9 * np.random.default_rng(args.seed).random(len(s))  # break ties at random
    rec["result"] = summarise(s, qc[m[ev]], qs[m[ev]])
    rec["seconds"] = round(time.time() - t0, 1)
    with open(ROOT / "results" / args.out / "rb.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
