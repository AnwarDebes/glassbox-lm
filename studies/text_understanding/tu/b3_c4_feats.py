#!/usr/bin/env python
"""Step 3b, the student's input bits (tu/PLAN_3b.md, "Student input"): existsT bits of K concepts fit on C4 tokens
under the a3B word table.

    python -I -B b3_c4_feats.py fit --sentences tu/a3/c4_sent_train_f.txt --out tu/b3/c4_concepts.npz [--K 512 --device cpu]
    python -I -B b3_c4_feats.py encode --concepts tu/b3/c4_concepts.npz --sentences FILE [--start 0 --limit N] --out X.npz
    python -I -B b3_c4_feats.py encode --concepts tu/b3/c4_concepts.npz --dataset clinc150 --split train --out X.npz

fit: the table's live rows read as w1/wordtab_feats.py reads a word table (a row whose codes are all zero has no word);
the tokens of every line of the sentence file (wordtab_feats.glove_tokens; a token with no row is dropped); the
distinct rows seen, in table order, L2-normalised; dl.boolean._spherical_kmeans with K concepts, 25 iterations and a
torch generator on --device seeded 0 (exists_tm.features' clustering, unchanged). Binarisation = the reader's existsT
rule (exists2_tm.py, kept in c3_exists_tm.py): per concept, dl.boolean.Thermo(8), i.e. the 7 quantiles (np.unique) of
the texts' max cosines to that concept, here over the first --thermo-rows lines of the sentence file (its order is
3a's random sample order). On the CPU the generator's stream differs from CUDA's, so CPU concepts (the P0 pilot's)
are not the GPU job's.
encode: per text, exists_tm.pooled's "exists" part (the max cosine of its tokens to each concept; a text with no token
gets 0.0), then the fitted per-concept thresholds (bit on when the max cosine is at or above the threshold), over the
whole live table, so any table word of any text counts. Output: np.packbits(bits, axis=1) as "X" with "n_features",
the row count and the input hashes.
--dataset reads dl.data.load(name).texts[split] only. Outputs are never overwritten.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(EMB / "w1"))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X
from dl.boolean import Thermo, _spherical_kmeans
from wordtab_feats import glove_tokens

TABLE = HERE / "a3" / "tables" / "a3B_d768.npz"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_table(path: Path) -> tuple[np.ndarray, list[str], dict]:
    z = np.load(path)
    M = z["codes"].astype(np.float32)
    words = [str(w) for w in z["words"]]
    live = np.abs(M).sum(1) > 0
    row = {w: i for i, w in enumerate(words) if live[i]}
    return M, words, row


def token_rows(texts, row: dict) -> tuple[list[list[int]], dict]:
    ids, n_tok, n_in, empty = [], 0, 0, 0
    for t in texts:
        toks = glove_tokens(t)
        r = [row[w] for w in toks if w in row]
        n_tok += len(toks)
        n_in += len(r)
        empty += not r
        ids.append(r)
    return ids, {"texts": len(ids), "oov_token_share": round(1 - n_in / max(1, n_tok), 5), "texts_without_token": empty}


def max_cos(M: np.ndarray, Cn: np.ndarray, ids: list[list[int]], device: str, bs: int = 1024) -> np.ndarray:
    dev = torch.device(device)
    E = torch.as_tensor(M, device=dev)
    En = torch.nn.functional.normalize(E, dim=1)
    C = torch.as_tensor(Cn, device=dev)
    return np.concatenate([X.pooled(ids[s : s + 8192], E, En, C, dev, bs=bs)["exists"] for s in range(0, len(ids), 8192)]) \
        if ids else np.zeros((0, Cn.shape[0]), np.float32)


def fit_concepts(M: np.ndarray, ids: list[list[int]], K: int, device: str, thermo_rows: int) -> dict:
    dev = torch.device(device)
    vocab = sorted({r for x in ids for r in x})
    Z = torch.nn.functional.normalize(torch.as_tensor(M[vocab], device=dev), dim=1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, K, 25, gen)
    a = (Z @ Cn.T).argmax(1).cpu().numpy()
    Cn = Cn.cpu().numpy().astype(np.float32)
    th = Thermo(8).fit(max_cos(M, Cn, ids[:thermo_rows], device)).th
    return {"Cn": Cn, "thresholds": [t.astype(np.float32).tolist() for t in th], "vocab_rows": len(vocab),
            "members": np.bincount(a, minlength=Cn.shape[0]).tolist(), "thermo_rows": min(thermo_rows, len(ids))}


def encode_bits(M: np.ndarray, Cn: np.ndarray, th: list, ids: list[list[int]], device: str, bs: int = 1024) -> np.ndarray:
    eb = X.ExistsBits(th, [str(k) for k in range(len(th))])  # the same transform as Thermo: (x_k >= t) per threshold
    out = [np.packbits(eb.transform(max_cos(M, Cn, ids[s : s + 8192], device, bs)), axis=1) for s in range(0, len(ids), 8192)]
    n_bits = sum(len(t) for t in th)
    return np.concatenate(out) if out else np.zeros((0, (n_bits + 7) // 8), np.uint8)


def read_lines(path: Path, start: int, limit: int) -> list[str]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit and len(out) >= limit:
                break
            out.append(line.rstrip("\n"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sentences", type=Path, required=True)
    f.add_argument("--table", type=Path, default=TABLE)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--K", type=int, default=512)
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--thermo-rows", type=int, default=200_000)
    f.add_argument("--device", default="cpu")
    e = sub.add_parser("encode")
    e.add_argument("--concepts", type=Path, required=True)
    e.add_argument("--table", type=Path, default=TABLE)
    e.add_argument("--sentences", type=Path)
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--dataset")
    e.add_argument("--split", default="train")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    M, _words, row = read_table(a.table)
    if a.cmd == "fit":
        texts = read_lines(a.sentences, 0, a.limit)
        ids, info = token_rows(texts, row)
        c = fit_concepts(M, ids, a.K, a.device, a.thermo_rows)
        meta = {"sentences": str(a.sentences), "sentences_sha16": sha16(a.sentences), "lines": len(texts), "limit": a.limit,
                "table": str(a.table), "table_sha16": sha16(a.table), "K": a.K, "device": a.device, **info,
                "vocab_rows": c["vocab_rows"], "n_bits": sum(len(t) for t in c["thresholds"]), "members": c["members"],
                "binarisation": f"existsT: dl.boolean.Thermo(8) on the max cosines of the first {c['thermo_rows']} lines",
                "b3_c4_feats_sha16": sha16(Path(__file__).resolve()), "exists_tm_sha16": sha16(EMB / "exists_tm.py"),
                "wordtab_feats_sha16": sha16(EMB / "w1" / "wordtab_feats.py"), "seconds": round(time.time() - t0, 1),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(a.out, "xb") as fo:
            np.savez(fo, Cn=c["Cn"], thresholds=json.dumps(c["thresholds"]), meta=json.dumps(meta))
        print(json.dumps({k: v for k, v in meta.items() if k != "members"}))
        return 0
    cz = np.load(a.concepts)
    th = json.loads(str(cz["thresholds"]))
    if a.dataset:
        sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
        from dl.data import load

        texts = load(a.dataset).texts[a.split]
        src = {"dataset": a.dataset, "split": a.split}
    else:
        texts = read_lines(a.sentences, a.start, a.limit)
        src = {"sentences": str(a.sentences), "sentences_sha16": sha16(a.sentences), "start": a.start, "limit": a.limit}
    ids, info = token_rows(texts, row)
    Xp = encode_bits(M, cz["Cn"], th, ids, a.device)
    n_bits = sum(len(t) for t in th)
    meta = {**src, "rows": len(texts), "n_features": n_bits, "concepts": str(a.concepts), "concepts_sha16": sha16(a.concepts),
            "table_sha16": sha16(a.table), "device": a.device, **info, "b3_c4_feats_sha16": sha16(Path(__file__).resolve()),
            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out, "xb") as fo:
        np.savez(fo, X=Xp, n_features=np.asarray(n_bits), meta=json.dumps(meta))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
