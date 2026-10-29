#!/usr/bin/env python
"""Step 3c follow-ups (a), (b), (b'): features for several arms of one set from ONE k-means (the integrity check's I1:
c3_feats.py ran once per arm, and its CUDA k-means is not deterministic, so HWU64's arms were not bit-paired).

    python -I -B c3x_feats.py --dataset clinc150 --table wordtab:TABLE.npz --arms orig2:none,twin:FILE[,F_rk:FILE] \
        --out-root tu/c3x/feats

tu/c3_feats.py's steps, unchanged except these:
- several arms in one process: the small table holds the words of every split and of every arm's texts, then one
  k-means (dl.boolean._spherical_kmeans, K 512, 25 iterations, CUDA generator seeded 0) on the ORIGINAL training
  tokens, one set of concept thresholds, and the original splits pooled once;
- each arm's cache = those original arrays, then the arm's kept texts pooled with the same table and concepts and
  appended to "train" (an arm "none" has no texts: the rebuilt originals);
- no equivalence check against a cache (every arm here shares one k-means by construction); instead each meta records
  the run id (the k-means centroids' sha256) and the sha256 of the concept names and of the thresholds, so a reader can
  check that its arms share them. The k-means runs on CUDA, as 3c's (not deterministic across runs; the comparisons need
  no determinism, since all arms of a set share one run).
Splits: train, val, oos_train, oos_val; never test or oos_test. Writes OUT_ROOT/<arm>/<set>__p32tok_K512.npz and
.meta.json. Refuses to overwrite.
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
import wordtab_feats as WF
from dl.boolean import _spherical_kmeans
from dl.data import load


def read_kept(path: str, ds, name: str) -> tuple[list[str], list[int], str | None]:
    if path == "none":
        return [], [], None
    kp = Path(path)
    texts, ys = [], []
    for line in kp.read_text().splitlines():
        c = json.loads(line)
        if c["set"] != name:
            continue
        assert ds.texts["train"][c["index"]] == c["source"] and ds.intents[c["y"]] == c["label"], "kept row does not match train"
        texts.append(c["text"])
        ys.append(int(c["y"]))
    return texts, ys, WF.sha256_file(kp)[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True, help="wordtab:PATH (a word table npz in w1's format)")
    ap.add_argument("--arms", required=True, help="arm:kept_file pairs, comma-separated; kept_file 'none' = originals only")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--device", default="cuda", help="cpu only for a functional test")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    arms = [tuple(x.split(":", 1)) for x in args.arms.split(",")]
    outs = {arm: (args.out_root / arm / f"{args.dataset}__p32tok_K{args.K}.npz",
                  args.out_root / arm / f"{args.dataset}__p32tok_K{args.K}.meta.json") for arm, _ in arms}
    for npz, mjs in outs.values():
        assert not npz.exists() and not mjs.exists(), f"{npz} exists"
    ds = load(args.dataset)
    splits = [sp for sp in X.SPLITS if sp in ds.texts]
    assert not ({"test", "oos_test"} & set(splits))
    aug = {arm: read_kept(path, ds, args.dataset) for arm, path in arms}
    assert args.table.startswith("wordtab:")
    TP = Path(args.table.split(":", 1)[1])
    zt = np.load(TP)
    live = np.abs(zt["codes"].astype(np.float32)).sum(1) > 0
    M = zt["codes"].astype(np.float32)
    words = [str(w) for w in zt["words"]]
    row = {w: i for i, w in enumerate(words) if live[i]}
    toks = {sp: [WF.glove_tokens(t) for t in ds.texts[sp]] for sp in splits}
    toks_aug = {arm: [WF.glove_tokens(t) for t in aug[arm][0]] for arm, _ in arms}
    need = sorted({row[w] for sp in splits for x in toks[sp] for w in x if w in row}
                  | {row[w] for arm, _ in arms for x in toks_aug[arm] for w in x if w in row})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(np.asarray(M[need]), device=dev)
    ids = {sp: [[local[row[w]] for w in x if w in row] for x in toks[sp]] for sp in splits}
    name_of = lambda i: words[need[int(i)]]
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, args.K, 25, gen)
    run_id = hashlib.sha256(Cn.cpu().numpy().astype(np.float32).tobytes()).hexdigest()[:16]
    sims = Z @ Cn.T
    a = sims.argmax(1)
    own = sims.gather(1, a[:, None])[:, 0].cpu().numpy()
    a = a.cpu().numpy()
    th = [np.unique(np.percentile(own[a == k], X.LEVELS)).astype(np.float32).tolist() if (a == k).any() else [2.0]
          for k in range(args.K)]
    near = sims.T.topk(min(4, len(vocab)), dim=1).indices.cpu().numpy()
    names = ["|".join(name_of(vocab[j]) for j in r) for r in near]
    base = {}
    for sp, x in ids.items():
        for k, v in X.pooled(x, E, En, Cn, dev).items():
            base[f"{sp}__{k}"] = v
    n_orig = len(ds.texts["train"])
    report = {}
    for arm, path in arms:
        texts, ys, ksha = aug[arm]
        out = dict(base)
        ids_aug = [[local[row[w]] for w in x if w in row] for x in toks_aug[arm]]
        if ids_aug:
            pa = X.pooled(ids_aug, E, En, Cn, dev)
            for k, v in pa.items():
                out[f"train__{k}"] = np.concatenate([base[f"train__{k}"], v])
        out["y_train"] = np.concatenate([ds.labels["train"], np.asarray(ys, np.int64)]).astype(ds.labels["train"].dtype)
        out["y_val"] = ds.labels["val"]
        out["n_orig"] = np.asarray(n_orig)
        meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": args.K,
                "members": np.bincount(a, minlength=args.K).tolist(), "dataset": args.dataset, "table": TP.stem,
                "table_sha256": WF.sha256_file(TP)[:16], "c3x_feats_sha256": WF.sha256_file(Path(__file__).resolve())[:16],
                "arm": arm, "n_orig": n_orig, "n_aug": len(texts), "kept": path, "kept_sha256": ksha,
                "kmeans_run_id": run_id, "concepts_sha16": hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16],
                "thresholds_sha16": hashlib.sha256(json.dumps(th).encode()).hexdigest()[:16], "arms_in_this_run": [x for x, _ in arms],
                "aug_sentences_without_token": int(sum(1 for x in ids_aug if not x))}
        npz, mjs = outs[arm]
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **out)
        mjs.write_text(json.dumps(meta))
        report[arm] = {"n_aug": len(texts), "npz_sha16": WF.sha256_file(npz)[:16]}
    print(json.dumps({"dataset": args.dataset, "kmeans_run_id": run_id, "arms": report, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
