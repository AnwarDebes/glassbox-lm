#!/usr/bin/env python
"""W1-S word control (EXPLORATORY, VAL only): glove42_feats.py with a word table built from C4 alone.

A copy of glove42_feats.py with one change and nothing else: the table choice wordtab:PATH reads a word table made by
w1/ppmi_svd2.py --stream words (npz with the word strings and their rows), in place of GloVe 42B; words that have
no row are dropped, as out-of-vocabulary GloVe words. Tokens (glove_feats.glove_tokens), k-means, pooling, splits and
outputs are unchanged. Docstring of glove42_feats.py:

N7 follow-up F2 (EXPLORATORY, VAL only): glove_feats.py (the N7 GloVe arm) with the GloVe 42B table on the intent sets.

A copy of glove_feats.py with two changes and nothing else: the table choice glove42B300 = GloVe 42B 300d (Common
Crawl, 42B tokens, 1.9M uncased words; the N7 table), read from the files glove42_prep.py parsed
(data_untrusted/glove42B/glove.42B.300d.vectors.npy and .words.json, sha256 in PARSED.sha256), memory-mapped; and
a word listed twice keeps its FIRST (most frequent) row, as n7_tm.py does for this table. Tokens, k-means,
pooling, splits and outputs are glove_feats.py's. Original docstring of glove_feats.py:

N7 GloVe arm (EXPLORATORY until registered; VAL only): the existsT features of test 2 with another static table.

    glove_feats.py --dataset clinc150 --table glove6B300 --out-dir DIR
    glove_feats.py --dataset clinc150 --table potion --out-dir DIR --check emb/clinc150__p32tok_K512.npz

The same steps as exists_tm.features (test 2), unchanged except the table and its tokenizer:
- potion: potion-base-32M (revision 1e5a03f8) and its own tokenizer, add_special_tokens=False. --check compares
  the result with test 2's cached features (the harness-equivalence check: the code path must give them again).
- glove6B300: GloVe 6B 300d (Pennington et al. 2014: a log-bilinear model fit to the co-occurrence counts of
  Wikipedia 2014 + Gigaword 5, 6B tokens, 400k lowercased words; no transformer at any stage; say "transformer-free",
  never "no neural network"), file data_untrusted/glove6B/glove.6B.300d.txt (sha256 in the meta), read as plain text. Tokens: the
  lowercased text, contractions split as the Stanford tokenizer does ("don't" -> "do n't"), then
  n't | 'xx | runs of letters and digits | any other single non-space character; a token not in GloVe is dropped.
Then, as test 2: the distinct tokens of the TRAINING texts (sorted by table row), L2-normalized, spherical k-means
(dl.boolean._spherical_kmeans, K 512, 25 iterations, CUDA generator seeded 0); per sentence and concept the max cosine
over its tokens (exists_tm.pooled; a sentence with no token gets 0.0); also the mean / max / min parts. Written as
DIR/{dataset}__p32tok_K512.npz + .meta.json, the file names exists2_tm.py reads, so the TM harness runs unchanged
(DIR/README.md says which table a folder holds). Splits: train, val, oos_train, oos_val; never test or oos_test.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent  # glassbox-lm/studies/text_understanding (this copy lives in w1/)
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from dl.boolean import _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402

GLOVE_TXT = HERE / "data_untrusted" / "glove6B" / "glove.6B.300d.txt"
GLOVE_NPY = HERE / "data_untrusted" / "glove6B" / "glove.6B.300d.vectors.npy"  # parsed once from the text file
GLOVE_WORDS = HERE / "data_untrusted" / "glove6B" / "glove.6B.300d.words.json"
CONTR = [(re.compile(r"n't\b"), " n't"), (re.compile(r"'(s|re|ve|ll|d|m)\b"), r" '\1")]
TOK = re.compile(r"n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def glove_tokens(text: str) -> list[str]:
    t = text.lower()
    for pat, rep in CONTR:
        t = pat.sub(rep, t)
    return TOK.findall(t)


def load_glove():
    if not GLOVE_NPY.exists():
        words, rows = [], []
        with open(GLOVE_TXT, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                words.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        M = np.stack(rows)
        assert M.shape == (400000, 300), M.shape
        np.save(GLOVE_NPY, M)
        GLOVE_WORDS.write_text(json.dumps(words))
    return np.load(GLOVE_NPY), json.loads(GLOVE_WORDS.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True, help="potion, or wordtab:PATH (a w1 word table npz)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--check", type=Path, default=None, help="potion only: test 2's cached features to compare with")
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.1)
    ds = load(args.dataset)
    splits = [sp for sp in X.SPLITS if sp in ds.texts]
    assert not ({"test", "oos_test"} & set(splits))
    if args.table == "potion":
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
        tok = st[0].tokenizer
        E = st[0].embedding.weight.detach().float().to(dev)
        ids = {sp: [e.ids for e in tok.encode_batch(ds.texts[sp], add_special_tokens=False)] for sp in splits}
        name_of = lambda i: tok.id_to_token(int(i))  # noqa: E731
        info = {"table": "potion-base-32M", "revision": X.REV}
    else:
        assert args.table.startswith("wordtab:")
        TP = Path(args.table.split(":", 1)[1])
        zt = np.load(TP)
        live = np.abs(zt["codes"].astype(np.float32)).sum(1) > 0
        M = zt["codes"].astype(np.float32)
        words = [str(w) for w in zt["words"]]
        row = {w: i for i, w in enumerate(words) if live[i]}  # each word appears once in a w1 word table
        toks = {sp: [glove_tokens(t) for t in ds.texts[sp]] for sp in splits}
        need = sorted({row[w] for sp in splits for x in toks[sp] for w in x if w in row})
        local = {g: j for j, g in enumerate(need)}  # GloVe row -> row of the small table (keeps GloVe's order)
        E = torch.as_tensor(np.asarray(M[need]), device=dev)
        ids = {sp: [[local[row[w]] for w in x if w in row] for x in toks[sp]] for sp in splits}
        n_tok = {sp: sum(len(x) for x in toks[sp]) for sp in splits}
        n_in = {sp: sum(1 for x in toks[sp] for w in x if w in row) for sp in splits}
        name_of = lambda i: words[need[int(i)]]  # noqa: E731
        info = {"table": TP.stem, "table_sha256": sha256_file(TP)[:16],
                "tokenizer": "lowercase, Stanford-style contraction split, regex " + TOK.pattern,
                "oov_token_share": {sp: round(1 - n_in[sp] / max(1, n_tok[sp]), 4) for sp in splits},
                "sentences_without_token": {sp: int(sum(1 for x in ids[sp] if not x)) for sp in splits}}
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, args.K, 25, gen)
    sims = Z @ Cn.T
    a = sims.argmax(1)
    own = sims.gather(1, a[:, None])[:, 0].cpu().numpy()
    a = a.cpu().numpy()
    th = [np.unique(np.percentile(own[a == k], X.LEVELS)).astype(np.float32).tolist() if (a == k).any() else [2.0]
          for k in range(args.K)]
    near = sims.T.topk(min(4, len(vocab)), dim=1).indices.cpu().numpy()
    names = ["|".join(name_of(vocab[j]) for j in r) for r in near]
    out = {}
    for sp, x in ids.items():
        for k, v in X.pooled(x, E, En, Cn, dev).items():
            out[f"{sp}__{k}"] = v
    out["y_train"], out["y_val"] = ds.labels["train"], ds.labels["val"]
    meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": args.K,
            "members": np.bincount(a, minlength=args.K).tolist(), "dataset": args.dataset, **info,
            "wordtab_feats_sha256": sha256_file(Path(__file__).resolve())[:16]}
    rep = {"dataset": args.dataset, "table": args.table, "train_vocab": len(vocab), "seconds": None}
    if args.check is not None:
        assert args.table == "potion"
        ref = np.load(args.check)
        rep["check"] = {k: float(np.abs(out[k] - ref[k]).max()) for k in sorted(out) if k in ref.files}
        rep["check_identical"] = all(np.array_equal(out[k], ref[k]) for k in sorted(out) if k in ref.files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{args.dataset}__p32tok_K{args.K}.npz", **out)
    (args.out_dir / f"{args.dataset}__p32tok_K{args.K}.meta.json").write_text(json.dumps(meta))
    rep["seconds"] = round(time.time() - t0, 1)
    rep.update({k: info[k] for k in ("oov_token_share", "sentences_without_token") if k in info})
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
