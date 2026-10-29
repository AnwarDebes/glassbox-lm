#!/usr/bin/env python
"""N7: existsT next to published Tsetlin machine text results (REGISTRATION_N7.md). One script, two steps:

    n7_tm.py --build --dataset trec|mr|r52|r8|sst2 --mode explore|test     # feature cache (GPU)
    n7_tm.py --dataset trec --mode explore|test --view VIEW --seed S       # one TM run (GPU)

Data (data_untrusted/, pinned Hugging Face revisions, SOURCE.txt in each folder; plain CSV / JSONL, read as text):
  trec  SetFit/TREC-QC, label_coarse (6 classes): published train 5,452, test 500
  mr    dxgp/MR (the PTE split): published train = train.csv + val.csv = 7,108, test 3,554
  r52   dxgp/R52 (the TextGCN split): published train = train.csv + val.csv = 6,532, test 2,568
  r8    dxgp/R8 (the TextGCN split): published train = train.csv + val.csv = 5,485, test 2,189
  sst2  SetFit/sst2: published train 6,920, test 1,821 (dev.jsonl not used)
Splits: the published train texts in file order, permuted by numpy default_rng(2026); the last 10% of that order is
our val split, the first 90% our train split (so we train on 90% of the published train set). --mode explore uses
train and val only and never opens the test file. --mode test (the registered run) adds the published test split.
Parts (fitted on our train split only, applied to every split):
  bow       binary bag of words: the 5,000 words with the highest train document frequency (ties: alphabetical);
            words = glove_feats.glove_tokens (lowercase, contraction split, regex).
  existsT_potion / existsT_glove
            the existsT encoder of test 2 (exists_tm.py) with potion-base-32M (its tokenizer) or GloVe 6B 300d
            (glove_feats.py words): spherical k-means (K 512, 25 iterations, CUDA generator seeded 0) of the distinct
            train tokens, per text and concept the max cosine over its tokens (exists_tm.pooled; no token: 0.0),
            Thermo(8) fitted on the train texts. A bit = "some word of the text has cos >= t to concept k".
Views: bow, existsT_potion, existsT_glove, bow+existsT_potion, bow+existsT_glove.
TM: the decision-layer CoTM (dl/tm.py, read only) with the settings of test 2 (4,000 clauses, T 4000, s 8, batch
128, hardest negative, negation, init half) except 100 epochs (test 2: 40); trained by the loop of CoTM.fit (same
batches, same random stream from the machine's own generator), but with no best-state restore and with val (and
in test mode test) scored after EVERY epoch. Statistics: (1) Yadav et al. 2021: the mean test accuracy over epochs
51-100 (a fixed statistic, nothing selected); (2) the single-read number: test accuracy at the epoch with the best
val accuracy (ties: the earliest epoch); also the epoch-100 accuracy. Rows: results_n7.jsonl (explore) or
results_n7_test.jsonl (test). Never prints a text.
"""

from __future__ import annotations

import argparse
import csv
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
import exists_tm as X  # noqa: E402
import glove_feats as GF  # noqa: E402
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

U = HERE / "data_untrusted"
CACHE = HERE / "emb_n7"
DATASETS = ("trec", "mr", "r52", "r8", "sst2")
VIEWS = ("bow", "existsT_potion", "existsT_glove", "bow+existsT_potion", "bow+existsT_glove")
N_BOW, K, EPOCHS, LAST = 5000, 512, 100, 50


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def read_csv(p: Path):
    csv.field_size_limit(10**9)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def read_jsonl(p: Path, label: str):
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return [r["text"] for r in rows], [int(r[label]) for r in rows]


def load(dataset: str, with_test: bool):
    if dataset == "trec":
        train = read_jsonl(U / "trec_setfit" / "train.jsonl", "label_coarse")
        get_test = lambda: read_jsonl(U / "trec_setfit" / "test.jsonl", "label_coarse")  # noqa: E731
    elif dataset == "sst2":
        train = read_jsonl(U / "sst2_setfit" / "train.jsonl", "label")
        get_test = lambda: read_jsonl(U / "sst2_setfit" / "test.jsonl", "label")  # noqa: E731
    else:
        d = U / {"mr": "mr_dxgp", "r52": "r52_dxgp", "r8": "r8_dxgp"}[dataset]
        a, b = read_csv(d / "train.csv"), read_csv(d / "val.csv")
        train = (a[0] + b[0], a[1] + b[1])
        get_test = lambda: read_csv(d / "test.csv")  # noqa: E731
    texts, labels = train
    perm = np.random.default_rng(2026).permutation(len(texts))
    cut = len(texts) - len(texts) // 10
    sp = {"train": ([texts[i] for i in perm[:cut]], [labels[i] for i in perm[:cut]]),
          "val": ([texts[i] for i in perm[cut:]], [labels[i] for i in perm[cut:]])}
    if with_test:
        sp["test"] = get_test()
    classes = sorted(set(labels))  # the published train labels
    idx = {c: i for i, c in enumerate(classes)}
    for k in sp:
        assert all(c in idx for c in sp[k][1]), f"a {k} label is not in the published train set"
    return {k: (t, np.array([idx[c] for c in y])) for k, (t, y) in sp.items()}, len(classes)


def exists_bits(table: str, texts: dict, dev: str, bs: int):
    if table == "potion":
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
        tok = st[0].tokenizer
        E = st[0].embedding.weight.detach().float().to(dev)
        ids = {k: [e.ids for e in tok.encode_batch(t, add_special_tokens=False)] for k, t in texts.items()}
        info = {"table": "potion-base-32M", "revision": X.REV}
    else:
        M, words = GF.load_glove()
        row = {w: i for i, w in enumerate(words)}
        toks = {k: [GF.glove_tokens(x) for x in t] for k, t in texts.items()}
        need = sorted({row[w] for t in toks.values() for x in t for w in x if w in row})
        local = {g: j for j, g in enumerate(need)}
        E = torch.as_tensor(M[need], device=dev)
        ids = {k: [[local[row[w]] for w in x if w in row] for x in t] for k, t in toks.items()}
        n_all = sum(len(x) for x in toks["train"])
        info = {"table": "glove.6B.300d", "oov_token_share_train": round(1 - sum(map(len, ids["train"])) / max(1, n_all), 4)}
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(En[torch.as_tensor(vocab, device=dev)], K, 25, gen)
    sims = {k: X.pooled(v, E, En, Cn, dev, bs=bs)["exists"] for k, v in ids.items()}
    th = Thermo(8).fit(sims["train"])
    bits = {k: th.transform(v).astype(np.uint8) for k, v in sims.items()}
    info.update({"train_vocab": len(vocab), "texts_without_token": {k: int(sum(1 for x in v if not x)) for k, v in ids.items()},
                 "tokens_per_text_median": {k: float(np.median([len(x) for x in v])) for k, v in ids.items()},
                 "on_share": {k: round(float(b.mean()), 4) for k, b in bits.items()}})
    return bits, info


def build(args) -> None:
    t0 = time.time()
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.1)
    sp, C = load(args.dataset, args.mode == "test")
    texts = {k: v[0] for k, v in sp.items()}
    df: dict = {}
    for t in texts["train"]:
        for w in set(GF.glove_tokens(t)):
            df[w] = df.get(w, 0) + 1
    vocab = sorted(df, key=lambda w: (-df[w], w))[:N_BOW]
    col = {w: j for j, w in enumerate(vocab)}

    def bow(ts):
        B = np.zeros((len(ts), len(vocab)), np.uint8)
        for i, t in enumerate(ts):
            for w in set(GF.glove_tokens(t)):
                if w in col:
                    B[i, col[w]] = 1
        return B

    out = {f"y_{k}": v[1] for k, v in sp.items()}
    info = {"dataset": args.dataset, "mode": args.mode, "n": {k: len(v[1]) for k, v in sp.items()}, "classes": C,
            "bow_words": len(vocab), "bow_min_df": int(df[vocab[-1]]), "bow_on_share": {}}
    for k, t in texts.items():
        out[f"bow_{k}"] = bow(t)
        info["bow_on_share"][k] = round(float(out[f"bow_{k}"].mean()), 4)
    bs = 64 if args.dataset in ("r52", "r8") else 1024
    info["pooled_bs"] = bs
    for table in ("potion", "glove"):
        bits, inf = exists_bits(table, texts, dev, bs)
        for k, b in bits.items():
            out[f"existsT_{table}_{k}"] = b
        info[table] = inf
    info["seconds"] = round(time.time() - t0, 1)
    info["n7_tm_sha256"] = sha16(Path(__file__).resolve())
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / f"{args.dataset}__{args.mode}.npz", **out)
    (CACHE / f"{args.dataset}__{args.mode}.meta.json").write_text(json.dumps(info))
    print(json.dumps(info))


def run(args) -> None:
    t0 = time.time()
    torch.cuda.set_per_process_memory_fraction(0.1)
    z = np.load(CACHE / f"{args.dataset}__{args.mode}.npz")
    meta = json.loads((CACHE / f"{args.dataset}__{args.mode}.meta.json").read_text())
    parts = args.view.split("+")
    evals = ("val", "test") if args.mode == "test" else ("val",)
    B = {k: np.concatenate([z[f"{p}_{k}"] for p in parts], 1) for k in ("train",) + evals}
    y = {k: z[f"y_{k}"] for k in ("train",) + evals}
    cfg = TMConfig(n_clauses=4000, T=args.T, s=args.s, epochs=EPOCHS, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=1)
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
    rec = {"registration": "REGISTRATION_N7.md", "dataset": args.dataset, "mode": args.mode, "view": args.view,
           "seed": args.seed, "n": meta["n"], "n_bits": int(B["train"].shape[1]),
           "on_share": {k: round(float(B[k].mean()), 4) for k in ("train",) + evals},
           "best_val_epoch": best_ep + 1, "val_at_best": curves["val"][best_ep],
           "val_mean_last50": round(float(np.mean(curves["val"][-LAST:])), 3), "val_final": curves["val"][-1]}
    if "test" in curves:
        rec.update({"test_mean_last50": round(float(np.mean(curves["test"][-LAST:])), 3),
                    "test_at_best_val": curves["test"][best_ep], "test_final": curves["test"][-1]})
    rec.update({"curves": curves, "cfg": asdict(cfg), "train_s": round(train_s, 1), "total_s": round(time.time() - t0, 1),
                "cache_meta": meta,
                "sha256": {"n7_tm.py": sha16(Path(__file__).resolve()), "glove_feats.py": sha16(HERE / "glove_feats.py"),
                           "exists_tm.py": sha16(HERE / "exists_tm.py")}})
    name = "results_n7_test.jsonl" if args.mode == "test" else "results_n7.jsonl"
    with open(HERE / name, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curves", "cfg", "cache_meta")}))


def main() -> None:
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--mode", required=True, choices=("explore", "test"))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--view", choices=VIEWS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="explore smoke only (timing); the registered runs use 100")
    ap.add_argument("--T", type=int, default=4000, help="vote margin (test 2: 4000)")
    ap.add_argument("--s", type=float, default=8.0, help="specificity (test 2: 8)")
    args = ap.parse_args()
    if args.mode == "test":
        assert args.epochs == 100
    EPOCHS = args.epochs
    if args.build:
        build(args)
    else:
        assert args.view, "--view is needed for a run"
        run(args)


if __name__ == "__main__":
    main()
