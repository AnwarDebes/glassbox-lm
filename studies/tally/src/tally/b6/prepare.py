"""B6 data preparation: splits and booleanized views for Fashion-MNIST and AG News.

Run once, with the preparation environment (it needs OpenCV):

    PYTHONPATH=src .venv-b6/bin/python -m tally.b6.prepare [--dataset fmnist|agnews|all]

Splits (fixed here, seeded, stratified by class, cut from the official training file):
    train   the rest of the official training file
    calib   held-out part of train: composite normalization, gate training, calibration check
    valid   tuning
    devtest the official test file (verdict, one look)

Views. Fashion-MNIST, the four specialists of Granmo 2023 (TMComposites) and the CAIR repository
scripts: adaptive Gaussian thresholding (block 11, C 2), 8-level thermometer (thresholds
(z+1)*255/9), HOG (window 28, block 12, stride 4, cell 4, 18 signed bins, bit = value >= 0.1).
These are fixed per image; nothing is fitted. AG News, three views after the CAIR IMDb team
script: binary word unigrams (chi2 top 5000), word bigrams (chi2 top 5000), char_wb 4-grams
(chi2 top 3000). Vocabulary and chi2 selection are fitted on the train split only.

Outputs go to data/b6/<dataset>/ (git-ignored) with manifest.json (sha256 of sources and
outputs, split sizes, package versions, duplicate counts).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STUDIES = os.path.dirname(STUDY)
HUB = os.path.join(STUDIES, "tm_language_model", "data", "hf_cache", "hub")
OUT = os.path.join(STUDY, "data", "b6")
SEED = 20261015

SOURCES = {
    "fmnist": {
        "train": os.path.join(HUB, "datasets--zalando-datasets--fashion_mnist", "snapshots",
                              "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2", "fashion_mnist",
                              "train-00000-of-00001.parquet"),
        "test": os.path.join(STUDY, "data", "fashion_mnist_hf", "test-00000-of-00001.parquet"),
    },
    "agnews": {
        "train": os.path.join(HUB, "datasets--fancyzhx--ag_news", "snapshots",
                              "eb185aade064a813bc0b7f42de02595523103ca4", "data",
                              "train-00000-of-00001.parquet"),
        "test": os.path.join(HUB, "datasets--fancyzhx--ag_news", "snapshots",
                             "eb185aade064a813bc0b7f42de02595523103ca4", "data",
                             "test-00000-of-00001.parquet"),
    },
}
SOURCE_URLS = {
    "fmnist": "https://huggingface.co/datasets/zalando-datasets/fashion_mnist (revision 531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2)",
    "agnews": "https://huggingface.co/datasets/fancyzhx/ag_news (revision eb185aade064a813bc0b7f42de02595523103ca4)",
}
# held-out sizes per class, cut from the official training file
PER_CLASS = {"fmnist": {"calib": 500, "valid": 500}, "agnews": {"calib": 1500, "valid": 1500}}
AG_VIEWS = {"word1": dict(analyzer="word", ngram_range=(1, 1), k=5000),
            "word2": dict(analyzer="word", ngram_range=(2, 2), k=5000),
            "char4": dict(analyzer="char_wb", ngram_range=(4, 4), k=3000)}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_indices(y, per_class, seed=SEED):
    """Stratified cut of the official training file into train, calib, valid."""
    rng = np.random.default_rng(seed)
    out = {"calib": [], "valid": [], "train": []}
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = per_class["calib"], per_class["calib"] + per_class["valid"]
        out["calib"].append(idx[:a])
        out["valid"].append(idx[a:b])
        out["train"].append(idx[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def save(dirname, name, arr, files):
    p = os.path.join(dirname, name)
    np.save(p, arr)
    files[name] = sha256(p)


def prepare_fmnist():
    import cv2
    import pyarrow.parquet as pq

    def load(p):
        t = pq.read_table(p).to_pydict()
        X = np.stack([cv2.imdecode(np.frombuffer(im["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
                      for im in t["image"]]).astype(np.uint8)
        return X, np.asarray(t["label"], dtype=np.uint32)

    Xtr, ytr = load(SOURCES["fmnist"]["train"])
    Xte, yte = load(SOURCES["fmnist"]["test"])
    assert Xtr.shape == (60000, 28, 28) and Xte.shape == (10000, 28, 28)
    sp = split_indices(ytr, PER_CLASS["fmnist"])
    hog = cv2.HOGDescriptor((28, 28), (12, 12), (4, 4), (4, 4), 18, 1, -1., 0, 0.2, True, 64, True)

    def views(X):
        th = np.empty_like(X)
        for i in range(X.shape[0]):
            th[i] = cv2.adaptiveThreshold(X[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thermo = np.stack([X >= (z + 1) * 255 / 9 for z in range(8)], axis=-1).astype(np.uint8)
        h = np.stack([hog.compute(X[i]).ravel() >= 0.1 for i in range(X.shape[0])]).astype(np.uint8)
        return {"thresh": th, "thermo": thermo, "hog": h, "raw": X.reshape(X.shape[0], -1)}

    d = os.path.join(OUT, "fmnist")
    os.makedirs(d, exist_ok=True)
    files = {}
    parts = {k: (Xtr[v], ytr[v], v) for k, v in sp.items()}
    parts["devtest"] = (Xte, yte, np.arange(len(yte)))
    for split, (X, y, idx) in parts.items():
        for v, arr in views(X).items():
            save(d, f"{v}_{split}.npy", arr, files)
        save(d, f"y_{split}.npy", y, files)
        save(d, f"idx_{split}.npy", idx.astype(np.int64), files)
    # exact pixel duplicates of dev-test images inside the train split (reported, not removed)
    seen = {Xtr[i].tobytes() for i in sp["train"]}
    dup = np.array([Xte[i].tobytes() in seen for i in range(len(Xte))])
    save(d, "devtest_dup_in_train.npy", dup, files)
    return d, files, {k: int(len(v[1])) for k, v in parts.items()}, {"devtest_exact_dup_in_train": int(dup.sum())}


def prepare_agnews():
    import pyarrow.parquet as pq
    import scipy.sparse as sps
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import SelectKBest, chi2

    def load(p):
        t = pq.read_table(p).to_pydict()
        return t["text"], np.asarray(t["label"], dtype=np.uint32)

    ttr, ytr = load(SOURCES["agnews"]["train"])
    tte, yte = load(SOURCES["agnews"]["test"])
    assert len(ttr) == 120000 and len(tte) == 7600
    sp = split_indices(ytr, PER_CLASS["agnews"])
    texts = {k: [ttr[i] for i in v] for k, v in sp.items()}
    labels = {k: ytr[v] for k, v in sp.items()}
    texts["devtest"], labels["devtest"] = tte, yte
    d = os.path.join(OUT, "agnews")
    os.makedirs(d, exist_ok=True)
    files, vocab = {}, {}
    for name, cfg in AG_VIEWS.items():
        vec = CountVectorizer(analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"], lowercase=True, binary=True)
        Xtr = vec.fit_transform(texts["train"])                      # train split only
        skb = SelectKBest(chi2, k=min(cfg["k"], Xtr.shape[1])).fit(Xtr, labels["train"])
        cols = skb.get_support(indices=True)
        names = vec.get_feature_names_out()[cols]
        vocab[name] = [str(x) for x in names]
        for split, tx in texts.items():
            X = (Xtr if split == "train" else vec.transform(tx))[:, cols].astype(np.uint8).tocsr()
            p = os.path.join(d, f"{name}_{split}.npz")
            sps.save_npz(p, X)
            files[os.path.basename(p)] = sha256(p)
    for split in texts:
        save(d, f"y_{split}.npy", labels[split], files)
        idx = sp[split] if split != "devtest" else np.arange(len(tte))
        save(d, f"idx_{split}.npy", np.asarray(idx, dtype=np.int64), files)
    with open(os.path.join(d, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    files["vocab.json"] = sha256(os.path.join(d, "vocab.json"))

    # exact duplicates (lowercased, whitespace collapsed) of dev-test texts inside the train split
    def norm(s):
        return " ".join(s.lower().split())
    seen = {norm(s) for s in texts["train"]}
    dup = np.array([norm(s) in seen for s in tte])
    save(d, "devtest_dup_in_train.npy", dup, files)
    dup_v = int(sum(norm(s) in seen for s in texts["valid"]))
    return d, files, {k: len(v) for k, v in labels.items()}, {"devtest_exact_dup_in_train": int(dup.sum()),
                                                              "valid_exact_dup_in_train": dup_v}


def versions():
    out = {}
    for m in ("numpy", "cv2", "pyarrow", "sklearn", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["fmnist", "agnews", "all"])
    a = ap.parse_args()
    for ds in (["fmnist", "agnews"] if a.dataset == "all" else [a.dataset]):
        t0 = time.time()
        d, files, sizes, dups = (prepare_fmnist if ds == "fmnist" else prepare_agnews)()
        man = {"dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source_url": SOURCE_URLS[ds],
               "sources": {k: {"path": os.path.relpath(p, STUDIES), "sha256": sha256(p)}
                           for k, p in SOURCES[ds].items()},
               "split_seed": SEED, "held_out_per_class": PER_CLASS[ds], "sizes": sizes,
               "duplicates": dups, "outputs": files, "versions": versions(),
               "wall_s": round(time.time() - t0, 1)}
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        print(json.dumps({"dataset": ds, "sizes": sizes, "duplicates": dups, "wall_s": man["wall_s"]}))


if __name__ == "__main__":
    main()
