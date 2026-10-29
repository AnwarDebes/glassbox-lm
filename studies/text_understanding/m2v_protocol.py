#!/usr/bin/env python
"""Registered run P1 (REGISTRATION_P1.md): the Model2Vec authors' low-data protocol, with the Tsetlin machine.

    m2v_protocol.py --dataset banking77|clinc_oos|massive_intent --draw 1 [--smoke]

The protocol of MinishLab/model2vec results/README.md ("Training Results"): 1,000 examples from the train set, the
full test set, weighted F1, no parameter tuned on a validation set. A draw is 1,000 examples drawn uniformly without
replacement (numpy default_rng(draw)) from the official training pool:
- banking77: the official train split in file order (10,003), test 3,080, 77 classes;
- clinc_oos: the "plus" configuration (clinc/oos-eval data_oos_plus.json, the source of HF clinc_oos "plus"):
  15,000 in-scope + 250 oos training queries, test 4,500 + 1,000 oos, 151 classes with oos as a class;
- massive_intent: MASSIVE 1.1 en-US train (11,514), test 2,974, 60 classes.
Everything fitted is fitted on the 1,000 drawn texts only: the 512 concepts (spherical k-means, seed 0, on the
distinct tokens of the 1,000 texts) and the thermometer thresholds (8 levels). Rows (one JSON line each):
- lr_mean: scikit-learn LogisticRegressionCV (defaults) on the normalized mean of the potion-base-32M token vectors
  (= StaticModel.encode, the model's normalize=true): the published model2vec + logreg recipe;
- lr_existsT: the same classifier on the existsT bits;
- tm_existsT, tm_mean+existsT, tm_mean: the decision-layer CoTM (dl/tm.py, read only; 4,000 clauses, T 4000, s 8,
  batch 128, hardest negative, negation, init half, seed = draw). Its epoch count is chosen on an inner hold-out (a
  random 20% of the 1,000, default_rng(1000 + draw); at most 120 epochs, checked every 4), then a fresh machine with
  the same seed is trained on all 1,000 for that many epochs and its final state is scored.
--smoke: 200 drawn examples, 4 epochs, scored on 500 undrawn training-pool examples; the test split is not read and
nothing is written. Otherwise rows are appended to results_p1.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exists_tm as X  # noqa: E402  (also puts the study's src on sys.path)
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import RAW, load  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

PUBLISHED = {"banking77": 79.95, "clinc_oos": 68.11, "massive_intent": 73.83}  # model2vec + logreg, weighted F1
PLUS = HERE / "data_untrusted" / "clinc_oos_plus" / "data_oos_plus.json"
PLUS_SHA16 = "bfcca9ae51562354"
ROWS = ("lr_mean", "lr_existsT", "tm_existsT", "tm_mean+existsT", "tm_mean")


def data(name: str):
    """Training pool and test split in a fixed order: texts, labels, number of classes."""
    if name == "banking77":
        tr, te = pd.read_parquet(RAW / "banking77_train.parquet"), pd.read_parquet(RAW / "banking77_test.parquet")
        return (tr["text"].tolist(), tr["label"].to_numpy().astype(np.int64),
                te["text"].tolist(), te["label"].to_numpy().astype(np.int64), 77)
    if name == "clinc_oos":
        assert hashlib.sha256(PLUS.read_bytes()).hexdigest()[:16] == PLUS_SHA16
        d = json.loads(PLUS.read_text())
        names = sorted({lab for sp in ("train", "test") for _, lab in d[sp]}) + ["oos"]
        idx = {k: i for i, k in enumerate(names)}
        tr, te = d["train"] + d["oos_train"], d["test"] + d["oos_test"]
        return ([t for t, _ in tr], np.array([idx[lab] for _, lab in tr], np.int64),
                [t for t, _ in te], np.array([idx[lab] for _, lab in te], np.int64), len(names))
    if name == "massive_intent":
        ds = load("massive_en")
        return ds.texts["train"], ds.labels["train"], ds.texts["test"], ds.labels["test"], len(ds.intents)
    raise ValueError(name)


def encode(tr_x, te_x, dev="cuda"):
    """mean and exists similarities of exists_tm.py, with the 512 concepts fitted on the drawn texts only."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
    tok = st[0].tokenizer
    E = st[0].embedding.weight.detach().float().to(dev)
    En = torch.nn.functional.normalize(E, dim=1)
    ids_tr = [e.ids for e in tok.encode_batch(tr_x, add_special_tokens=False)]
    ids_te = [e.ids for e in tok.encode_batch(te_x, add_special_tokens=False)]
    vocab = sorted({t for x in ids_tr for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, 512, 25, gen)
    return X.pooled(ids_tr, E, En, Cn, dev), X.pooled(ids_te, E, En, Cn, dev), len(vocab)


def lr_cv(Xtr, ytr, Xte):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegressionCV

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = LogisticRegressionCV(n_jobs=1).fit(Xtr, ytr)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return clf.predict(Xte), {"C": float(np.atleast_1d(clf.C_)[0]), "convergence_warnings": n_conv}


def tm_cfg(epochs: int, seed: int, every: int) -> TMConfig:
    return TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=epochs, batch_size=128, neg_sampling="hardest", negation=True,
                    max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                    half=True, seed=seed, eval_every=every)


def tm_row(Btr, ytr, Bte, C, draw, max_epochs, every):
    perm = np.random.default_rng(1000 + draw).permutation(len(ytr))
    ho, fit = perm[: len(ytr) // 5], perm[len(ytr) // 5 :]
    tm = make_tm(C, Btr.shape[1], tm_cfg(max_epochs, draw, every), "half")
    tm.fit(Btr[fit], ytr[fit], Btr[ho], ytr[ho])
    epochs, ho_acc, s1 = int(tm.best_epoch), float(tm.best_val), tm.train_seconds
    del tm
    tm = make_tm(C, Btr.shape[1], tm_cfg(epochs, draw, every), "half")  # fresh machine, all examples
    tm.fit(Btr, ytr)
    pred = tm.predict(torch.as_tensor(Bte, dtype=torch.uint8, device="cuda")).cpu().numpy()
    return pred, {"epochs": epochs, "holdout_acc": round(ho_acc, 4), "train_s": round(s1 + tm.train_seconds, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(PUBLISHED))
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--mem", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    from sklearn.metrics import f1_score

    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.mem)
    t0 = time.time()
    pool_x, pool_y, te_x, te_y, C = data(args.dataset)
    n = 200 if args.smoke else 1000
    idx = np.random.default_rng(args.draw).choice(len(pool_x), n, replace=False)
    if args.smoke:  # scored on undrawn pool examples: the test split is not read
        rest = np.setdiff1d(np.arange(len(pool_x)), idx)[:500]
        te_x, te_y = [pool_x[i] for i in rest], pool_y[rest]
    tr_x, tr_y = [pool_x[i] for i in idx], pool_y[idx]
    Ftr, Fte, n_vocab = encode(tr_x, te_x)
    th_m, th_e = Thermo(8).fit(Ftr["mean"]), Thermo(8).fit(Ftr["exists"])
    B = {"mean": (th_m.transform(Ftr["mean"]), th_m.transform(Fte["mean"])),
         "existsT": (th_e.transform(Ftr["exists"]), th_e.transform(Fte["exists"]))}
    B["mean+existsT"] = tuple(np.concatenate([B["mean"][i], B["existsT"][i]], 1) for i in range(2))
    for row in args.rows.split(","):
        t1 = time.time()
        kind, view = row.split("_", 1)
        if kind == "lr":
            Xtr, Xte = (Ftr["mean"], Fte["mean"]) if view == "mean" else (B[view][0].astype(np.float32), B[view][1].astype(np.float32))
            pred, extra = lr_cv(Xtr, tr_y, Xte)
        else:
            Xtr = B[view][0]
            pred, extra = tm_row(Xtr, tr_y, B[view][1], C, args.draw, 4 if args.smoke else 120, 2 if args.smoke else 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(te_y, pred, average="weighted")
        rec = {"dataset": args.dataset, "draw": args.draw, "row": row, "n_train": n, "n_test": int(len(te_y)),
               "n_classes": C, "classes_in_draw": int(len(np.unique(tr_y))), "train_vocab": n_vocab,
               "n_features": int(Xtr.shape[1]), "weighted_f1": round(100 * float(f1), 2),
               "acc": round(100 * float((pred == te_y).mean()), 2), **extra, "published_lr_mean": PUBLISHED[args.dataset],
               "row_s": round(time.time() - t1, 1), "total_s": round(time.time() - t0, 1)}
        if args.smoke:
            rec["smoke"] = True
        else:
            with open(HERE / "results_p1.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
