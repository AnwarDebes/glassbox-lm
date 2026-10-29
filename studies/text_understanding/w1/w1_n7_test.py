#!/usr/bin/env python
"""W1-T (REGISTRATION_W1T.md): the word-control arm on N7's test splits (the second use of those splits, declared).

    w1_n7_test.py --build --dataset trec      # feats_n7_test/ppw_S1000_d500_e0.5/{ds}.npz: train / val / test bits
    w1_n7_test.py --dataset trec --seed 1     # one TM run: val and test scored after every epoch

Table: tables/ppw_S1000_d500_e0.5.npz (sha256 in the registration; built by ppmi_svd2.py --stream words at S1000).
Texts: n7_tm.load(dataset, True) = our train / carved val split of the published train set and the published test.
Bits: w1_n7w.bits_for_words (words by glove_feats.glove_tokens, looked up in the table; k-means of the train tokens,
K 512, 25 iterations, CUDA generator seeded 0; max cosine over the text's tokens with N7's batch size; Thermo(8) fitted
on train). At build the train and val bits must equal the val-run cache feats_n7/ppw_S1000_d500_e0.5/{ds}.npz
exactly (the same encoder); the build stops otherwise.
TM: n7_tm.py's run loop in test mode (dl/tm.py CoTM, 4,000 clauses, T 80, s 9, 100 epochs, batch 128, hardest
negative, negation, init half; val and test after every epoch, no best-state restore). Statistics as N7: S2 = test
accuracy at the best val epoch (ties: the earliest), S1 = mean test accuracy over epochs 51-100, epoch-100 test.
Rows in w1/results_w1t_test.jsonl.
"""

from __future__ import annotations

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
sys.path.insert(0, str(HERE))
import w1_n7 as P  # noqa: E402
import w1_n7w as PW  # noqa: E402

TABLE = HERE / "tables" / "ppw_S1000_d500_e0.5.npz"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=P.N.DATASETS)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.1)
    out_dir = HERE / "feats_n7_test" / TABLE.stem
    if args.build:
        sp, C = P.N.load(args.dataset, True)
        assert set(sp) == {"train", "val", "test"}
        texts = {k: v[0] for k, v in sp.items()}
        bs = 64 if args.dataset in ("r52", "r8") else 1024
        bits, info = PW.bits_for_words(TABLE, texts, dev, bs)
        ref = np.load(HERE / "feats_n7" / TABLE.stem / f"{args.dataset}.npz")
        same = {k: bool(np.array_equal(bits[k], ref[f"existsT_{k}"])) for k in ("train", "val")}
        assert all(same.values()), f"train / val bits differ from the val-run cache: {same}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{args.dataset}.npz", **{f"existsT_{k}": b for k, b in bits.items()},
                            **{f"y_{k}": v[1] for k, v in sp.items()})
        info.update({"dataset": args.dataset, "classes": C, "n": {k: len(v[1]) for k, v in sp.items()}, "pooled_bs": bs,
                     "train_val_bits_equal_val_run_cache": same, "w1_n7_test_sha256": sha16(Path(__file__).resolve()),
                     "seconds": round(time.time() - t0, 1)})
        (out_dir / f"{args.dataset}.meta.json").write_text(json.dumps(info))
        print(json.dumps(info))
        return
    z = np.load(out_dir / f"{args.dataset}.npz")
    meta = json.loads((out_dir / f"{args.dataset}.meta.json").read_text())
    assert meta["table_sha256"] == sha16(TABLE) and all(meta["train_val_bits_equal_val_run_cache"].values())
    B = {k: z[f"existsT_{k}"] for k in ("train", "val", "test")}
    y = {k: z[f"y_{k}"] for k in ("train", "val", "test")}
    cfg = P.TMConfig(n_clauses=4000, T=80, s=9.0, epochs=P.N.EPOCHS, batch_size=128, neg_sampling="hardest", negation=True,
                     max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                     half=True, seed=args.seed, eval_every=1)
    tm = P.make_tm(int(meta["classes"]), B["train"].shape[1], cfg, "half")
    Xt = torch.as_tensor(np.ascontiguousarray(B["train"]), dtype=torch.uint8, device=tm.dev)
    yt = torch.as_tensor(y["train"], dtype=torch.long, device=tm.dev)
    Xe = {k: torch.as_tensor(np.ascontiguousarray(B[k]), dtype=torch.uint8, device=tm.dev) for k in ("val", "test")}
    ye = {k: torch.as_tensor(y[k], dtype=torch.long, device=tm.dev) for k in ("val", "test")}
    curves = {"val": [], "test": []}
    n, t1 = Xt.shape[0], time.time()
    for epoch in range(cfg.epochs):  # n7_tm.run's loop (CoTM.fit's batches and random stream), scoring every epoch
        perm = torch.randperm(n, generator=tm.gen, device=tm.dev)
        for start in range(0, n, cfg.batch_size):
            bidx = perm[start : start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
        for k in ("val", "test"):
            curves[k].append(round(float((tm.predict(Xe[k]) == ye[k]).float().mean()) * 100, 2))
    torch.cuda.synchronize()
    best = int(np.argmax(curves["val"]))
    rec = {"registration": "REGISTRATION_W1T.md", "dataset": args.dataset, "mode": "test", "view": "existsT_c4words",
           "table": TABLE.stem, "table_sha256": meta["table_sha256"], "seed": args.seed, "n": meta["n"],
           "n_bits": int(B["train"].shape[1]), "on_share": meta["on_share"], "best_val_epoch": best + 1,
           "val_at_best": curves["val"][best], "val_mean_last50": round(float(np.mean(curves["val"][-P.N.LAST:])), 3),
           "test_at_best_val": curves["test"][best], "test_mean_last50": round(float(np.mean(curves["test"][-P.N.LAST:])), 3),
           "test_final": curves["test"][-1], "curves": curves, "cfg": asdict(cfg), "train_s": round(time.time() - t1, 1),
           "cache_meta": meta, "sha256": {"w1_n7_test.py": sha16(Path(__file__).resolve()), "w1_n7.py": sha16(HERE / "w1_n7.py"),
                                          "w1_n7w.py": sha16(HERE / "w1_n7w.py"), "n7_tm.py": sha16(HERE.parent / "n7_tm.py")}}
    with open(HERE / "results_w1t_test.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curves", "cfg", "cache_meta")}))


if __name__ == "__main__":
    main()
