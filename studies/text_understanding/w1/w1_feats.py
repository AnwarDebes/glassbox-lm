#!/usr/bin/env python
"""W1: the existsT features of test 2 with a word table keyed by potion-base-32M token ids (VAL only).

    w1_feats.py --dataset clinc150 --table tables/NAME.npz --out-dir feats/NAME
    w1_feats.py --dataset clinc150 --table potion --out-dir feats/potion_check --check ../emb/clinc150__p32tok_K512.npz

The steps of glove_feats.py's potion path (= exists_tm.features of test 2), unchanged except the table:
- tokens: the potion-base-32M tokenizer (revision 1e5a03f8), add_special_tokens=False, so every table sees exactly
  the tokens potion sees; a token whose row is missing or all zero in the table is dropped (as an out-of-vocabulary
  word of GloVe);
- --table potion: potion's own 63,091 rows through this code path; with --check the result is compared with test 2's
  cached features (the harness-equivalence check);
- then as test 2: the distinct tokens of the TRAINING texts (sorted by token id), L2-normalized, spherical k-means
  (dl.boolean._spherical_kmeans, K 512, 25 iterations, CUDA generator seeded 0); per sentence and concept the max
  cosine over its tokens (exists_tm.pooled; a sentence with no token gets 0.0); also the mean / max / min parts.
Written as DIR/{dataset}__p32tok_K512.npz + .meta.json, the names exists_tm.features reads. Splits: train, val,
oos_train, oos_val; never test or oos_test.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from dl.boolean import _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402

TOK_JSON = Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots") / X.REV / "tokenizer.json"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True, help="tables/NAME.npz, or potion")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--check", type=Path, default=None)
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
        full = st[0].embedding.weight.detach().float().numpy()
        vocab_ids, codes = np.arange(full.shape[0]), full
        info = {"table": "potion-base-32M (through w1_feats.py)", "revision": X.REV}
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(TOK_JSON))
        p = (HERE / args.table) if not Path(args.table).is_absolute() else Path(args.table)
        z = np.load(p)
        vocab_ids, codes = z["vocab_ids"].astype(np.int64), z["codes"].astype(np.float32)
        info = {"table": p.stem, "table_path": str(p), "table_sha256": sha16(p)}
    has = np.abs(codes).sum(1) > 0
    row_of = {int(t): j for j, t in enumerate(vocab_ids) if has[j]}
    raw = {sp: [e.ids for e in tok.encode_batch(ds.texts[sp], add_special_tokens=False)] for sp in splits}
    need = sorted({t for sp in splits for x in raw[sp] for t in x if t in row_of})
    local = {t: j for j, t in enumerate(need)}  # token id -> row of the small table (keeps token-id order)
    E = torch.as_tensor(codes[[row_of[t] for t in need]], device=dev)
    ids = {sp: [[local[t] for t in x if t in local] for x in raw[sp]] for sp in splits}
    n_tok = {sp: sum(len(x) for x in raw[sp]) for sp in splits}
    n_in = {sp: sum(len(x) for x in ids[sp]) for sp in splits}
    info.update({"token_share_without_vector": {sp: round(1 - n_in[sp] / max(1, n_tok[sp]), 4) for sp in splits},
                 "sentences_without_token": {sp: int(sum(1 for x in ids[sp] if not x)) for sp in splits}})
    name_of = lambda i: tok.id_to_token(int(need[int(i)]))  # noqa: E731
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
            "w1_feats_sha256": sha16(Path(__file__).resolve())}
    rep = {"dataset": args.dataset, "table": info["table"], "train_vocab": len(vocab), **{k: info[k] for k in
           ("token_share_without_vector", "sentences_without_token")}}
    if args.check is not None:
        ref = np.load(args.check)
        rep["check_max_abs_diff"] = {k: float(np.abs(out[k] - ref[k]).max()) for k in sorted(out) if k in ref.files}
        rep["check_identical"] = all(np.array_equal(out[k], ref[k]) for k in sorted(out) if k in ref.files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{args.dataset}__p32tok_K{args.K}.npz", **out)
    (args.out_dir / f"{args.dataset}__p32tok_K{args.K}.meta.json").write_text(json.dumps(meta))
    rep["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
