#!/usr/bin/env python
"""Registered confirmatory run (TEST split, used once): see REGISTRATION_T2.md.

    confirm_tm.py --dataset clinc150 --view mean|mean+existsT|existsT|minilm --seed 1 --emb-dir DIR [--smoke]

Views, pooling and TM settings are those of exists_tm.py (imported from it, unchanged). The features are
recomputed from scratch with the test splits included (own cache file), so the concepts are refit (k-means
seed 0 on the distinct training tokens; GPU sums can differ in the last bits from the exploratory cache).
minilm = thermo:8 on the decision-layer study's cached all-MiniLM-L6-v2 embeddings (reference row).
The epoch is chosen on val, as in every exploratory row; test accuracy is read once, at that epoch.
Appends one JSON line to results_confirm.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exists_tm as X  # noqa: E402  (also puts the study's src on sys.path)
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

ALL = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
MINI = Path(f"{_REPO_ROOT}/studies/decision_layer/data/emb")


def features(dataset: str, K: int, emb_dir: Path, dev: str):
    path, meta_path = emb_dir / f"{dataset}__p32tok_K{K}_confirm.npz", emb_dir / f"{dataset}__p32tok_K{K}_confirm.meta.json"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, json.loads(meta_path.read_text())
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
    tok = st[0].tokenizer
    E = st[0].embedding.weight.detach().float().to(dev)
    En = torch.nn.functional.normalize(E, dim=1)
    ds = load(dataset)
    ids = {sp: [e.ids for e in tok.encode_batch(ds.texts[sp], add_special_tokens=False)] for sp in ALL if sp in ds.texts}
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, K, 25, gen)
    sims = Z @ Cn.T
    a = sims.argmax(1)
    own = sims.gather(1, a[:, None])[:, 0].cpu().numpy()
    a = a.cpu().numpy()
    th = [np.unique(np.percentile(own[a == k], X.LEVELS)).astype(np.float32).tolist() if (a == k).any() else [2.0]
          for k in range(K)]
    near = sims.T.topk(min(4, len(vocab)), dim=1).indices.cpu().numpy()
    names = ["|".join(tok.id_to_token(vocab[j]) for j in row) for row in near]
    out = {}
    for sp, x in ids.items():
        for k, v in X.pooled(x, E, En, Cn, dev).items():
            out[f"{sp}__{k}"] = v
    for sp in ("train", "val", "test"):
        out[f"y_{sp}"] = ds.labels[sp]
    meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": K}
    np.savez(path, **out)
    meta_path.write_text(json.dumps(meta))
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, no test evaluation, nothing written")
    args = ap.parse_args()

    t0 = time.time()
    ds = load(args.dataset)
    C = len(ds.intents)
    if args.view == "minilm":
        z = np.load(MINI / f"{args.dataset}__minilm.npz")
        th = Thermo(8).fit(z["train"])
        B = {sp: th.transform(z[sp]) for sp in ("train", "val", "test")}
        y = {sp: z[f"y_{sp}"] for sp in ("train", "val", "test")}
    else:
        F, meta = features(args.dataset, args.K, args.emb_dir, "cuda")
        for sp in ALL:  # existsT: the exists similarities with thermometer bits (exists2_tm.py)
            if f"{sp}__exists" in F:
                F[f"{sp}__existsT"] = F[f"{sp}__exists"]
        view = X.View(args.view, meta).fit(F)
        B = {sp: view.transform(F, sp) for sp in ("train", "val", "test")}
        y = {sp: F[f"y_{sp}"] for sp in ("train", "val", "test")}
    for sp in ("train", "val", "test"):
        assert (y[sp] == ds.labels[sp]).all()
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=2)
    if args.smoke:
        cfg.epochs = 1
    tm = make_tm(C, B["train"].shape[1], cfg, "half")
    tm.fit(B["train"], y["train"], B["val"], y["val"])
    if args.smoke:
        print(json.dumps({"smoke": True, "view": args.view, "n_bits": int(B["train"].shape[1]), "val_acc_1_epoch": round(tm.best_val, 4)}))
        return
    pt = tm.predict(torch.as_tensor(B["test"], dtype=torch.uint8, device="cuda")).cpu().numpy()
    rec = {"tag": args.tag, "dataset": args.dataset, "view": args.view, "seed": args.seed, "n_bits": int(B["train"].shape[1]),
           "on_share": round(float(B["train"].mean()), 3),
           "cfg": asdict(cfg), "val_acc": round(tm.best_val, 4), "best_epoch": getattr(tm, "best_epoch", None),
           "test_acc": round(float((pt == y["test"]).mean()), 4), "n_test": int(len(pt)),
           "train_s": round(tm.train_seconds, 1), "total_s": round(time.time() - t0, 1)}
    with open(HERE / "results_confirm.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
