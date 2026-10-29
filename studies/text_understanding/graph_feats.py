#!/usr/bin/env python
"""Node symbols for the Graph Tsetlin Machine test (test 3, LOG.md): one node per token, meaning bits per node.

    graph_feats.py --dataset clinc150 --emb-dir DIR

Same table and concepts as test 2 (exists_tm.py): potion-base-32M (pinned revision), K = 512 spherical
k-means concepts (seed 0) of the distinct training tokens. Same thresholds as existsT: thermometer quantiles
(8 levels) of the sentence-level max similarity per concept over the training sentences. A node (token) gets
the bit "cos(token, concept k) >= t" for every concept k and threshold t. The OR of these node bits over the
tokens of a sentence is exactly the existsT bit vector of the sentence; this is checked on every train
sentence. Writes {dataset}__graph_K{K}.npz and .meta.json into DIR. Test texts are tokenized, never scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exists_tm as X  # noqa: E402  (puts the decision-layer src on sys.path; read only)
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--emb-dir", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
    tok = st[0].tokenizer
    E = st[0].embedding.weight.detach().float().to(dev)
    En = torch.nn.functional.normalize(E, dim=1)
    ds = load(args.dataset)
    ids = {sp: [e.ids for e in tok.encode_batch(ds.texts[sp], add_special_tokens=False)] for sp in SPLITS}
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, args.K, 25, gen)
    near = (Z @ Cn.T).T.topk(min(4, len(vocab)), dim=1).indices.cpu().numpy()
    names = ["|".join(tok.id_to_token(vocab[j]) for j in row) for row in near]

    used = np.array(sorted({t for sp in SPLITS for x in ids[sp] for t in x}), dtype=np.int64)
    S = (En[torch.as_tensor(used, device=dev)] @ Cn.T).cpu().numpy()  # [U, K] cos(token, concept)
    row = {int(t): i for i, t in enumerate(used)}
    # sentence-level max over tokens (0 for an empty sentence, as pooled() in exists_tm.py)
    mx_train = np.stack([S[[row[t] for t in x]].max(0) if x else np.zeros(args.K, np.float32) for x in ids["train"]])
    th = Thermo(8).fit(mx_train).th
    bits = np.concatenate([(S[:, k : k + 1] >= t[None, :]) for k, t in enumerate(th)], 1)  # [U, n_sym]
    n_sym = bits.shape[1]

    # check: OR over a sentence's node bits == existsT bits of the sentence (thermometer of the max)
    flat = Thermo(8)
    flat.th = th
    ref = flat.transform(mx_train)
    ok = all(np.array_equal(bits[[row[t] for t in x]].any(0), ref[i].astype(bool)) for i, x in enumerate(ids["train"]) if x)

    out = {"used": used, "node_bits": np.packbits(bits, axis=1), "n_sym": np.array(n_sym),
           "th_flat": np.concatenate(th).astype(np.float32), "th_off": np.cumsum([0] + [len(t) for t in th]),
           "train_vocab": np.array(vocab, dtype=np.int64)}
    for sp in SPLITS:
        out[f"{sp}__tok"] = np.array([t for x in ids[sp] for t in x], dtype=np.int64)
        out[f"{sp}__off"] = np.cumsum([0] + [len(x) for x in ids[sp]]).astype(np.int64)
        out[f"y_{sp}"] = ds.labels[sp]
    path = args.emb_dir / f"{args.dataset}__graph_K{args.K}.npz"
    np.savez(path, **out)
    lens = np.diff(out["train__off"])
    meta = {"dataset": args.dataset, "K": args.K, "n_sym": int(n_sym), "concepts": names,
            "tokens": {int(t): tok.id_to_token(int(t)) for t in used},
            "or_of_nodes_equals_existsT_on_train": bool(ok), "train_vocab": len(vocab), "used_tokens": int(len(used)),
            "node_on_share": round(float(bits.mean()), 4),
            "train_tokens_per_sentence": {"mean": round(float(lens.mean()), 2), "max": int(lens.max()),
                                          "empty": int((lens == 0).sum())},
            "n": {sp: len(ids[sp]) for sp in SPLITS}}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("concepts", "tokens")}))


if __name__ == "__main__":
    main()
