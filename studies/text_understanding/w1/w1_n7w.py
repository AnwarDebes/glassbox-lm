#!/usr/bin/env python
"""W1-S word control on suite 2 (N7 sets, CARVED VAL only): w1_n7.py with word tokens in place of potion tokens.

    w1_n7w.py --build --dataset trec --table tables/NAME.npz
    w1_n7w.py --dataset trec --table tables/NAME.npz --seed 1

Everything is w1_n7.py's (splits, k-means, pooling, Thermo(8), TM setting, statistics, files) except the tokens: words
by glove_feats.glove_tokens (as n7_tm.py's GloVe branch), looked up as strings in a word table made by
w1/ppmi_svd2.py --stream words; a word without a row is dropped. No --check (there is no earlier word-table cache).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w1_n7 as P  # noqa: E402


def bits_for_words(table: Path, texts: dict, dev: str, bs: int):
    z = np.load(table)
    codes = z["codes"].astype(np.float32)
    words = [str(w) for w in z["words"]]
    live = np.abs(codes).sum(1) > 0
    row = {w: i for i, w in enumerate(words) if live[i]}
    toks = {k: [P.N.GF.glove_tokens(x) for x in t] for k, t in texts.items()}
    need = sorted({row[w] for t in toks.values() for x in t for w in x if w in row})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(codes[need], device=dev)
    ids = {k: [[local[row[w]] for w in x if w in row] for x in t] for k, t in toks.items()}
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = P._spherical_kmeans(En[torch.as_tensor(vocab, device=dev)], P.N.K, 25, gen)
    sims = {k: P.X.pooled(v, E, En, Cn, dev, bs=bs)["exists"] for k, v in ids.items()}
    th = P.Thermo(8).fit(sims["train"])
    bits = {k: th.transform(v).astype(np.uint8) for k, v in sims.items()}
    n_tok = {k: sum(len(x) for x in v) for k, v in toks.items()}
    info = {"table": table.stem, "table_sha256": P.sha16(table), "tokens": "glove_feats.glove_tokens (words)",
            "train_vocab": len(vocab),
            "token_share_without_vector": {k: round(1 - sum(map(len, ids[k])) / max(1, n_tok[k]), 4) for k in ids},
            "texts_without_token": {k: int(sum(1 for x in v if not x)) for k, v in ids.items()},
            "on_share": {k: round(float(b.mean()), 4) for k, b in bits.items()},
            "w1_n7w_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    return bits, info


if __name__ == "__main__":
    assert "--check" not in sys.argv, "no --check for word tables"
    P.bits_for = bits_for_words
    P.main()
