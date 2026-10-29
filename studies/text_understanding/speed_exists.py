#!/usr/bin/env python
"""Rough CPU speed, one thread, text -> TM input bits, CLINC150 val (3,000 queries). Written for test 2;
thresholds switched to the POST HOC existsT variant before its first run (the speed does not depend on them).

- mean:   tokenize, average the token vectors, normalize, thermometer (7 thresholds per dimension).
- exists: tokenize, look up each token's precomputed packed bit code (existsT, about 3,500 bits), bitwise OR.
Concepts are recomputed here on CPU (k-means seed 0 on the distinct training tokens), so the codes are the
same kind as in exists_tm.py, not bit-identical. Also checks that OR of per-token codes equals "max cosine,
then threshold" on every val query. Rough: the machine is shared; read ratios, not absolute times.
Writes speed_exists.json here.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(1)
LEVELS = (0, 10, 25, 50, 75, 90, 100)
st = SentenceTransformer("minishlab/potion-base-32M", device="cpu", revision="1e5a03f8eeb2c98b928fbbd846f22f816360919f")
tok = st[0].tokenizer
E = st[0].embedding.weight.detach().float().numpy()
En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
ds = load("clinc150")
train, texts = ds.texts["train"], ds.texts["val"]

# offline part (once): concepts, thresholds, per-token codes for the whole table
ids_tr = [e.ids for e in tok.encode_batch(train, add_special_tokens=False)]
vocab = sorted({t for x in ids_tr for t in x})
gen = torch.Generator()
gen.manual_seed(0)
Cn = _spherical_kmeans(torch.as_tensor(En[vocab]), 512, 25, gen).numpy()
S_all = En @ Cn.T  # [V, K]
# existsT thresholds (exists2_tm.py): thermometer quantiles of the max similarity over the training sentences
ms_tr = np.stack([S_all[x].max(0) if x else np.zeros(512, np.float32) for x in ids_tr])
th = Thermo(8).fit(ms_tr).th
codes = np.concatenate([(S_all[:, k : k + 1] >= t[None, :]) for k, t in enumerate(th)], 1)
packed = np.packbits(codes, axis=1)  # [V, n_bits / 8]
mean_tr = np.stack([E[x].mean(0) if x else np.zeros(E.shape[1], np.float32) for x in ids_tr])
mean_tr /= np.maximum(np.linalg.norm(mean_tr, axis=1, keepdims=True), 1e-12)
thermo = Thermo(8).fit(mean_tr)
T = np.stack([np.pad(t, (0, 7 - len(t)), constant_values=np.inf) for t in thermo.th])  # [d, 7]


def enc_mean(batch):
    ids = [e.ids for e in tok.encode_batch(batch, add_special_tokens=False)]
    m = np.stack([E[x].mean(0) for x in ids])
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return np.packbits((m[:, :, None] >= T[None]).reshape(len(batch), -1), axis=1)


def enc_exists(batch):
    ids = [e.ids for e in tok.encode_batch(batch, add_special_tokens=False)]
    return np.stack([np.bitwise_or.reduce(packed[x], axis=0) for x in ids])


def enc_tokenize(batch):
    return [e.ids for e in tok.encode_batch(batch, add_special_tokens=False)]


# identity check: OR of codes == threshold of the max cosine
ids_va = enc_tokenize(texts)
ok = all(np.array_equal(np.unpackbits(np.bitwise_or.reduce(packed[x], axis=0))[: codes.shape[1]],
                        np.concatenate([(S_all[x, k].max() >= t) for k, t in enumerate(th)]).astype(np.uint8))
         for x in ids_va[:500])
out = {"or_equals_max_then_threshold_first_500": bool(ok), "exists_bits": int(codes.shape[1]),
       "table_bytes": int(packed.nbytes), "vocab_rows": int(packed.shape[0])}
for name, f in (("tokenize_only", enc_tokenize), ("mean_thermo", enc_mean), ("exists_or", enc_exists)):
    f(texts[:64])
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for s in range(0, len(texts), 64):
            f(texts[s : s + 64])
        best = min(best, time.perf_counter() - t0)
    out[name] = {"seconds": round(best, 4), "queries_per_s": round(len(texts) / best, 1)}
    print(name, out[name], flush=True)
print(out)
Path(__file__).with_name("speed_exists.json").write_text(json.dumps(out, indent=1))
