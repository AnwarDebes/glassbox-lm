#!/usr/bin/env python
"""N9 (EXPLORATORY, carved val only; plan LOG.md 12:49:05Z, rules in LOG.md before any run): the combined view's own
setting search, the fair re-test of "the meaning bits add over bag of words". A copy of n8_sweep.py (ae15f29b21e9b687);
the only changes: this docstring head, the GPU memory cap (25% of the card instead of 10%, so that literal-budget runs
fit), the view part existsT_c4, the option --check-n8, the study label and the rows file results_n9_sweep.jsonl.

    n9_sweep.py --dataset trec --stage n9s1 [--view bow] [--seed 1] [--clauses 4000] [--T 80] [--s 9] [--negation 1]
                [--batch 128] [--neg hardest] [--max-literals 0] [--check-n7] [--check-n8]

Data: N7's explore caches (emb_n7/{ds}__explore.npz; train and carved val only, the test split is never opened); the
view part existsT_c4 = W1-S's C4-word bits for N7's carved splits (w1/feats_n7/ppw_S1000_d500_e0.5/{ds}.npz, made by
w1_n7w.py from n7_tm.load's carve), used only when their label arrays equal the explore cache's, row for row.
--check-n8: the val curve must equal the val curve of N8's row with the same set, view, seed and setting
(results_n8_sweep.jsonl), else exit 3.
Loop: n7_tm.py's run loop, copied unchanged (the loop of CoTM.fit, val scored after every epoch, no best-state
restore); only the TMConfig fields come from the options. --check-n7: the val curve must equal the val curve of N7's
registered test run with the same set, view and seed (results_n7_test.jsonl; the N7 setting only), else exit 3.
Statistics as N7: S1-val = mean val accuracy over epochs 51-100, best val (ties: the earliest epoch).
One row per run in results_n8_sweep.jsonl. Never prints a text.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DL / "src"))
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

CACHE = HERE / "emb_n7"
W1F = HERE / "w1" / "feats_n7" / "ppw_S1000_d500_e0.5"
DATASETS = ("trec", "mr", "r52", "r8", "sst2")
EPOCHS, LAST = 100, 50


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--view", default="bow")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--s", type=float, default=9.0)
    ap.add_argument("--negation", type=int, default=1, choices=(0, 1))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--neg", default="hardest")
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--check-n7", action="store_true")
    ap.add_argument("--check-n8", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    torch.cuda.set_per_process_memory_fraction(0.25)
    z = np.load(CACHE / f"{args.dataset}__explore.npz")
    meta = json.loads((CACHE / f"{args.dataset}__explore.meta.json").read_text())
    parts = args.view.split("+")
    evals = ("val",)
    y = {k: z[f"y_{k}"] for k in ("train",) + evals}
    c4 = None
    if "existsT_c4" in parts:
        c4 = W1F / f"{args.dataset}.npz"
        zc = np.load(c4)
        assert all(np.array_equal(zc[f"y_{k}"], y[k]) for k in ("train",) + evals), "C4-word rows differ from N7's"
        z = {**{k: z[k] for k in z.files}, **{f"existsT_c4_{k}": zc[f"existsT_{k}"] for k in ("train",) + evals}}
    B = {k: np.concatenate([z[f"{p}_{k}"] for p in parts], 1) for k in ("train",) + evals}
    cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, epochs=EPOCHS, batch_size=args.batch, neg_sampling=args.neg,
                   negation=bool(args.negation), max_literals=args.max_literals, budget_mode="exact",
                   budget_select="threshold", max_negated=0, clause_drop=0.0, half=True, seed=args.seed, eval_every=1)
    tm = make_tm(int(meta["classes"]), B["train"].shape[1], cfg, "half")
    dev = tm.dev
    Xt = torch.as_tensor(np.ascontiguousarray(B["train"]), dtype=torch.uint8, device=dev)
    yt = torch.as_tensor(y["train"], dtype=torch.long, device=dev)
    Xe = {k: torch.as_tensor(np.ascontiguousarray(B[k]), dtype=torch.uint8, device=dev) for k in evals}
    ye = {k: torch.as_tensor(y[k], dtype=torch.long, device=dev) for k in evals}
    curves = {k: [] for k in evals}
    n = Xt.shape[0]
    t1 = time.time()
    for epoch in range(cfg.epochs):  # the loop of CoTM.fit, scoring every epoch, no best-state restore
        perm = torch.randperm(n, generator=tm.gen, device=dev)
        for start in range(0, n, cfg.batch_size):
            bidx = perm[start : start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
        for k in evals:
            curves[k].append(round(float((tm.predict(Xe[k]) == ye[k]).float().mean()) * 100, 2))
    torch.cuda.synchronize()
    train_s = time.time() - t1
    best_ep = int(np.argmax(curves["val"]))  # earliest best val epoch (0-based)
    rec = {"study": "N9 (exploratory, carved val)", "stage": args.stage, "dataset": args.dataset, "view": args.view,
           "seed": args.seed, "setting": {"clauses": args.clauses, "T": args.T, "s": args.s, "negation": args.negation,
                                          "batch": args.batch, "neg": args.neg, "max_literals": args.max_literals},
           "n": meta["n"], "n_bits": int(B["train"].shape[1]), "best_val_epoch": best_ep + 1,
           "val_at_best": curves["val"][best_ep], "val_mean_last50": round(float(np.mean(curves["val"][-LAST:])), 3),
           "val_final": curves["val"][-1], "curves": curves, "cfg": asdict(cfg), "train_s": round(train_s, 1),
           "total_s": round(time.time() - t0, 1),
           "cache_sha256": sha16(CACHE / f"{args.dataset}__explore.npz"), "c4_cache_sha256": sha16(c4) if c4 else None,
           "sha256": {"n9_sweep.py": sha16(Path(__file__).resolve()), "dl/tm.py": sha16(DL / "src/dl/tm.py"),
                      "dl/tm_init.py": sha16(DL / "src/dl/tm_init.py")}}
    if args.check_n7:
        ref = [json.loads(x) for x in (HERE / "results_n7_test.jsonl").read_text().splitlines() if x.strip()]
        ref = [r for r in ref if r["dataset"] == args.dataset and r["view"] == args.view and int(r["seed"]) == args.seed]
        same = bool(ref) and ref[0]["curves"]["val"] == curves["val"]
        rec["check_n7_val_curve_equal"] = same
    if args.check_n8:
        ref = [json.loads(x) for x in (HERE / "results_n8_sweep.jsonl").read_text().splitlines() if x.strip()]
        ref = [r for r in ref if r["dataset"] == args.dataset and r["view"] == args.view and int(r["seed"]) == args.seed
               and r["setting"] == rec["setting"]]
        rec["check_n8_val_curve_equal"] = bool(ref) and ref[0]["curves"]["val"] == curves["val"]
    with open(HERE / "results_n9_sweep.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curves", "cfg", "n")}), flush=True)
    if args.check_n7 and not rec["check_n7_val_curve_equal"]:
        sys.exit(3)
    if args.check_n8 and not rec["check_n8_val_curve_equal"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
