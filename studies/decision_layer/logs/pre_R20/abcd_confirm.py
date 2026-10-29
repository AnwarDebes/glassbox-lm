#!/usr/bin/env python
"""Registered ABCD next-action runs: one arm, test split scored once per run.

    abcd_confirm.py --registration R9 --name tm_full --arm tm --parts <parts> --seeds 1 2

Arms: ``tm`` (readable bits; best-dev snapshot), ``lr`` (logistic regression on the same bits, C
from the dev grid), ``mlp`` (one hidden layer on the same bits, best dev epoch), ``lr_raw`` and
``mlp_raw`` (raw embeddings of cust, recent and full plus the memory bits), ``rule`` (nearest
next-button centre of the last 3 turns), ``table`` (most frequent next button given nearest
subflow and last action). Writes results/confirm/<registration>/<name>/seed<k>.npz (pred, y,
convo_id, ast weights; class sums for the TM) and seed<k>.json (dev and test accuracy, settings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples, memory_from_rows  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.mlp import fit_mlp, mlp_scores  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import INITS, make_tm  # noqa: E402

GRID_C = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def cap_gpu() -> None:
    """Opt-in cap on this process's share of the shared GPU: only when DL_GPU_FRAC is set (unset: no cap,
    as in every run through R18). Added 2026-10-17 for the no-network runs."""
    frac = os.environ.get("DL_GPU_FRAC")
    if frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))


def bits_sha256(X: np.ndarray) -> str:
    """SHA-256 of a bit matrix (uint8, C order): equal bits give equal hashes across runs."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def peak_gpu_mb() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True, choices=["tm", "lr", "mlp", "lr_raw", "mlp_raw", "rule", "table"])
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star"], help="star: scripts/star_prepare.py examples (added 2026-10-17 14:18)")
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--lex-k", type=int, default=1000)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--init", default="half", choices=INITS)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true", help="code check: dev stands in for test, output in results/explore/dryrun")
    args = ap.parse_args()
    cap_gpu()

    out = (ROOT / "results" / "explore" / "dryrun" / args.name) if args.dry_run else (ROOT / "results" / "confirm" / args.registration / args.name)
    out.mkdir(parents=True, exist_ok=True)
    tr, blist = load_examples("train", args.dataset)
    dv, _ = load_examples("dev", args.dataset)
    te, _ = load_examples("dev" if args.dry_run else "test", args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    em["test"] = em["dev"] if args.dry_run else {f: E[f"test__{f}"] for f in ("cust", "recent", "full")}
    y = {s: np.array([r["y"] for r in rows]) for s, rows in (("train", tr), ("dev", dv), ("test", te))}
    C_ = len(blist)
    parts = tuple(p for p in args.parts.split(",") if p)
    meta_te = {"y": y["test"], "convo_id": np.array([r["convo_id"] for r in te]), "k": np.array([r["k"] for r in te]),
               "w_bert": np.array([r["ast_count_bert"] for r in te]), "w_roberta": np.array([r["ast_count_roberta"] for r in te])}

    meta_bits: dict = {}  # bits_sha256 and static_info, filled once the view is built

    def save(seed: int, pred_te: np.ndarray, pred_dv: np.ndarray, info: dict, S_te=None) -> None:
        arrays = {"pred": pred_te, **meta_te}
        if S_te is not None:
            arrays["S"] = S_te.astype(np.int32)
        np.savez_compressed(out / f"seed{seed}.npz", **arrays)
        rec = {"registration": args.registration, "name": args.name, "arm": args.arm, "parts": parts, "encoder": args.encoder,
               "seed": seed, "dev_acc": float((pred_dv == y["dev"]).mean()), "test_acc": float((pred_te == y["test"]).mean()),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}
        rec.update(meta_bits)
        rec["peak_gpu_mb"] = peak_gpu_mb()
        (out / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "stats")}), flush=True)

    todo = [s for s in args.seeds if not (out / f"seed{s}.json").exists()]
    if not todo:
        print("all seeds exist")
        return

    if args.arm in ("lr_raw", "mlp_raw"):
        # the same memory as the readable view: done, last, count, and the action before last
        def mem_all(rows):
            M2 = np.zeros((len(rows), len(blist)), dtype=np.float32)
            for i, r in enumerate(rows):
                if len(r["prev"]) >= 2:
                    M2[i, blist.index(r["prev"][-2])] = 1
            return np.concatenate([memory_from_rows(rows, blist).astype(np.float32), M2], 1)

        R = {s: np.concatenate([em[s]["full"], em[s]["cust"], em[s]["recent"], mem_all(rows)], 1)
             for s, rows in (("train", tr), ("dev", dv), ("test", te))}
        for seed in todo:
            if args.arm == "lr_raw":
                best = None
                for c in GRID_C:
                    W, b = fit_logreg(R["train"], y["train"], C_, c)
                    acc = float((logits(R["dev"], W, b).argmax(1).cpu().numpy() == y["dev"]).mean())
                    if best is None or acc > best[0]:
                        best = (acc, c, W, b)
                _, c, W, b = best
                save(seed, logits(R["test"], W, b).argmax(1).cpu().numpy(), logits(R["dev"], W, b).argmax(1).cpu().numpy(), {"C": c})
            else:
                net, info = fit_mlp(R["train"], y["train"], R["dev"], y["dev"], C_, seed=seed)
                save(seed, mlp_scores(net, R["test"]).argmax(1).cpu().numpy(), mlp_scores(net, R["dev"]).argmax(1).cpu().numpy(), info)
        return

    view = ABCDView(parts=parts, lex_k=args.lex_k, seed=0).fit(tr, em["train"], blist)
    X = {s: view.transform(rows, em[s]) for s, rows in (("train", tr), ("dev", dv), ("test", te))}
    meta_bits["bits_sha256"] = {s: bits_sha256(X[s]) for s in ("train", "dev", "test")}
    meta_bits["static_info"] = view.static_info()
    (out / "bit_names.json").write_text(json.dumps(view.names))
    for seed in todo:
        if args.arm == "tm":
            cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.epochs, max_literals=args.max_literals,
                           budget_select="rank", seed=seed)
            tm = make_tm(C_, X["train"].shape[1], cfg, init=args.init)
            tm.fit(X["train"], y["train"], X["dev"], y["dev"])
            S_te = tm.class_sums(torch.as_tensor(X["test"], dtype=torch.uint8, device="cuda")).cpu().numpy()
            S_dv = tm.class_sums(torch.as_tensor(X["dev"], dtype=torch.uint8, device="cuda")).cpu().numpy()
            st = tm.state()
            np.savez_compressed(out / f"model_seed{seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"])
            save(seed, S_te.argmax(1), S_dv.argmax(1), {"cfg": asdict(cfg), "init": args.init, "best_epoch": getattr(tm, "best_epoch", None),
                                                         "train_s": round(tm.train_seconds, 1), "stats": tm.stats(),
                                                         "n_bits": int(X["train"].shape[1])}, S_te=S_te)
        elif args.arm == "lr":
            best = None
            for c in GRID_C:
                W, b = fit_logreg(X["train"], y["train"], C_, c)
                acc = float((logits(X["dev"], W, b).argmax(1).cpu().numpy() == y["dev"]).mean())
                if best is None or acc > best[0]:
                    best = (acc, c, W, b)
            _, c, W, b = best
            np.savez_compressed(out / f"lr_weights_seed{seed}.npz", W=W.cpu().numpy(), b=b.cpu().numpy())
            save(seed, logits(X["test"], W, b).argmax(1).cpu().numpy(), logits(X["dev"], W, b).argmax(1).cpu().numpy(),
                 {"C": c, "n_bits": int(X["train"].shape[1])})
        elif args.arm == "mlp":
            net, info = fit_mlp(X["train"], y["train"], X["dev"], y["dev"], C_, seed=seed)
            save(seed, mlp_scores(net, X["test"]).argmax(1).cpu().numpy(), mlp_scores(net, X["dev"]).argmax(1).cpu().numpy(),
                 {**info, "n_bits": int(X["train"].shape[1])})
        elif args.arm == "rule":
            p_te = view.step.sims(em["test"]["recent"]).argmax(1).cpu().numpy()
            p_dv = view.step.sims(em["dev"]["recent"]).argmax(1).cpu().numpy()
            save(seed, p_te, p_dv, {})
        elif args.arm == "table":
            sub = {s: view.intent.sims(em[s]["cust"]).argmax(1).cpu().numpy() for s in ("train", "dev", "test")}
            t2, t1 = defaultdict(Counter), defaultdict(Counter)
            glob = Counter(y["train"].tolist())
            for r, s_ in zip(tr, sub["train"]):
                last = r["prev"][-1] if r["prev"] else "-"
                t2[(int(s_), last)][r["y"]] += 1
                t1[last][r["y"]] += 1

            def tab(rows, ss):
                out_ = []
                for r, s_ in zip(rows, ss):
                    last = r["prev"][-1] if r["prev"] else "-"
                    c = t2.get((int(s_), last)) or t1.get(last) or glob
                    out_.append(c.most_common(1)[0][0])
                return np.array(out_)

            save(seed, tab(te, sub["test"]), tab(dv, sub["dev"]), {})


if __name__ == "__main__":
    main()
