#!/usr/bin/env python
"""Step 3d (VAL only; tu/PLAN_3d.md "Readers"): the existsT features of one arm and one set, from a table that may hold
pair rows.

    python -I -B d3_feats.py --dataset clinc150 --arm J --table tu/d3/tables/d3J_d768.npz --pair-tokens on \
        --kmeans all --out-root tu/d3/feats

tu/m1_feats.py (bb93c24f72edfe67) with these changes and nothing else:
(1) the splits are train, val, oos_train, oos_val (exists_tm.SPLITS; never test or oos_test), and no potion path;
(2) one arm per run, with no paraphrase rows (the originals only);
(3) a text's tokens are its live words (as before), then, with --pair-tokens on and a table that has pair rows
    (d3_fit.py's format), its adjacent word pairs that the table lists (d3_pairs.pair_keys), as rows after the words;
(4) --kmeans words clusters only the word types of the training split; --kmeans all clusters every training token
    type, words and pairs (the same as words for a table without pairs);
(5) the meta records the pair settings and the pair coverage of every split (descriptive); it keeps m1_feats.py's
    fields except those of paraphrase rows and of the test split ("kept", "kept_sha256", "arms_in_this_run",
    "aug_sentences_without_token", "test_sentences_without_token"), names this file's hash "d3_feats_sha256" and adds
    "val_sentences_without_token";
(6) --pair-tokens on with a table that has no pair rows is refused (no arm loses its pairs silently);
(7) --kmeans-seed seeds the k-means generator (default 0, as before; other values only for the descriptive k-means
    spread arms), recorded in the meta;
(8) the GPU memory cap is 0.25 of the card (m1_feats.py: 0.1), since the pair rows enlarge the token table.
The k-means (dl.boolean._spherical_kmeans, K 512, 25 iterations) and the thresholds are fit on the training split's
token types, as before. With a table without pair rows and seed 0 this gives tu/m1_feats.py's train and val arrays.
Writes OUT_ROOT/<arm>/<set>__p32tok_K512.npz, .centroids.npy, then .meta.json last (the chain skips a build only when
its meta exists). Refuses to overwrite.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X
import wordtab_feats as WF
from dl.boolean import _spherical_kmeans

from d3_pairs import pair_keys
from m1_feats import load_ds

SPLITS = X.SPLITS  # train, val, oos_train, oos_val: never test, never oos_test


class PairTable:
    """A word table in w1's format, with d3_fit.py's optional pair rows (row V + j for pair j)."""

    def __init__(self, path: Path, pair_tokens: bool):
        zt = np.load(path)
        codes = zt["codes"].astype(np.float32)
        live = np.abs(codes).sum(1) > 0
        self.words = [str(w) for w in zt["words"]]
        self.V = len(self.words)
        self.row = {w: i for i, w in enumerate(self.words) if live[i]}
        self.has_pairs = "pair_codes" in zt.files
        self.pair_tokens = bool(pair_tokens and self.has_pairs)
        self.n_pairs = len(zt["pair_codes"]) if self.has_pairs else 0
        if self.pair_tokens:
            self.M = np.concatenate([codes, zt["pair_codes"].astype(np.float32)])
            w1, w2 = zt["pair_w1"].astype(np.int64), zt["pair_w2"].astype(np.int64)
            self.pair_row = {int(a) * self.V + int(b): j for j, (a, b) in enumerate(zip(w1, w2))}
            self.pair_name = [f"{self.words[int(a)]}_{self.words[int(b)]}" for a, b in zip(w1, w2)]
        else:
            self.M, self.pair_row, self.pair_name = codes, {}, []
        self.name, self.sha = path.stem, WF.sha256_file(path)[:16]

    def ids(self, texts: list[str]) -> list[list[int]]:
        out = []
        for t in texts:
            toks = WF.glove_tokens(t)
            w = [self.row[x] for x in toks if x in self.row]
            p = [self.V + self.pair_row[k] for k in pair_keys(toks, self.row, self.V) if k in self.pair_row]
            out.append(w + p)
        return out

    def token(self, i: int) -> str:
        return self.words[i] if i < self.V else self.pair_name[i - self.V]

    def coverage(self, texts: list[str]) -> dict:
        """Descriptive: adjacent live-word pairs of the texts, and how many the table lists."""
        n_adj = n_listed = n_with = 0
        for t in texts:
            ks = pair_keys(WF.glove_tokens(t), self.row, self.V)
            hit = sum(k in self.pair_row for k in ks)
            n_adj += len(ks)
            n_listed += hit
            n_with += hit > 0
        return {"texts": len(texts), "adjacent_word_pairs": n_adj, "listed": n_listed,
                "listed_share": round(n_listed / max(1, n_adj), 4), "texts_with_a_pair_token": n_with}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--pair-tokens", choices=("on", "off"), required=True)
    ap.add_argument("--kmeans", choices=("all", "words"), required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--kmeans-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cpu only for a functional test")
    ap.add_argument("--load", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    npz = args.out_root / args.arm / f"{args.dataset}__p32tok_K{args.K}.npz"
    mjs, cnp = npz.with_suffix(".meta.json"), npz.with_suffix(".centroids.npy")
    for p in (npz, mjs, cnp):
        assert not p.exists(), f"{p} exists"
    ds = load_ds(args.dataset, args.load)
    splits = [sp for sp in SPLITS if sp in ds.texts]
    assert not ({"test", "oos_test"} & set(splits))
    tab = PairTable(args.table, args.pair_tokens == "on")
    assert tab.pair_tokens or args.pair_tokens == "off", f"--pair-tokens on, but {args.table} has no pair rows"
    gids = {sp: tab.ids(ds.texts[sp]) for sp in splits}
    need = sorted({g for sp in splits for x in gids[sp] for g in x})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(np.asarray(tab.M[need]), device=dev)
    ids = {sp: [[local[g] for g in x] for x in gids[sp]] for sp in splits}
    name_of = lambda i: tab.token(need[int(i)])
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x if args.kmeans == "all" or need[t] < tab.V})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.kmeans_seed)
    Cn = _spherical_kmeans(Z, args.K, 25, gen)
    cents = Cn.cpu().numpy().astype(np.float32)
    run_id = hashlib.sha256(cents.tobytes()).hexdigest()[:16]
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
    n_orig = len(ds.texts["train"])
    out["y_train"] = ds.labels["train"]
    out["y_val"] = ds.labels["val"]
    out["n_orig"] = np.asarray(n_orig)
    is_pair = np.asarray([need[t] >= tab.V for t in vocab])
    pair_concepts = int(sum(any(need[vocab[j]] >= tab.V for j in r) for r in near))
    meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": args.K,
            "members": np.bincount(a, minlength=args.K).tolist(), "dataset": args.dataset, "table": tab.name,
            "table_sha256": tab.sha, "d3_feats_sha256": WF.sha256_file(Path(__file__).resolve())[:16], "arm": args.arm,
            "n_orig": n_orig, "n_aug": 0, "splits": splits, "kmeans_run_id": run_id,
            "concepts_sha16": hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16],
            "thresholds_sha16": hashlib.sha256(json.dumps(th).encode()).hexdigest()[:16],
            "pair_tokens": tab.pair_tokens, "kmeans_types": args.kmeans, "kmeans_seed": args.kmeans_seed,
            "table_pairs": tab.n_pairs,
            "train_vocab_words": int((~is_pair).sum()), "train_vocab_pairs": int(is_pair.sum()),
            "concepts_with_a_pair_among_top4": pair_concepts,
            "coverage": {sp: tab.coverage(ds.texts[sp]) for sp in splits} if tab.pair_tokens else {},
            "val_sentences_without_token": int(sum(1 for x in ids["val"] if not x))}
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **out)
    np.save(cnp, cents)
    mjs.write_text(json.dumps(meta))
    print(json.dumps({"dataset": args.dataset, "arm": args.arm, "table": tab.name, "kmeans_run_id": run_id,
                      "train_vocab_words": meta["train_vocab_words"], "train_vocab_pairs": meta["train_vocab_pairs"],
                      "npz_sha16": WF.sha256_file(npz)[:16], "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
