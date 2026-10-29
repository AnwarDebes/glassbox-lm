#!/usr/bin/env python
"""Step 3c, the features (VAL only; tu/PLAN_3c.md "Features and readers"): w1/wordtab_feats.py's word-table path
(001a93b259a1b34f), run again for the original splits, with the kept paraphrases pooled by the same table and the same
512 concepts and appended to "train".

    python -I c3_feats.py --dataset clinc150 --table wordtab:TABLE.npz --kept tu/c3/kept_F.jsonl --arm F \
        --out-dir tu/c3/feats/F_ppw_S1000 --ref-dir w1/feats/ppw_S1000_d500_e0.5

The same tokens (glove_tokens), table rows, k-means (dl.boolean._spherical_kmeans, K 512, 25 iterations, CUDA
generator seeded 0), concept thresholds and pooling (exists_tm.pooled) as wordtab_feats.py. Two additions only:
- the k-means vocabulary stays the ORIGINAL training tokens: a paraphrase's words are pooled, but they do not move the
  concepts (the small table also holds the paraphrases' words; that changes no vector of an original word);
- "train__*" = the originals (dl.data order), then this set's kept paraphrases (the kept file's order); y_train = their
  labels (a paraphrase has its source's label); "n_orig" = the number of originals (c3_exists_tm.py and c3_lr.py fit
  the thermometer on those rows only).
Equivalence check: before the paraphrases are appended, every array of the original splits is compared with the
cache in --ref-dir (the largest absolute difference, and whether all are identical); the concept names and thresholds
are compared with its meta. Recorded in the meta and printed. With --kept none the cache holds the rebuilt originals
only (arm "orig"). Splits: train, val, oos_train, oos_val; never test or oos_test. Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
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
import exists_tm as X  # noqa: E402
import wordtab_feats as WF  # noqa: E402
from dl.boolean import _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True, help="wordtab:PATH (a word table npz in w1's format)")
    ap.add_argument("--kept", required=True, help="a kept_*.jsonl of c3_filter.py, or none")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--device", default="cuda", help="cpu only for a functional test (the check then cannot match)")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    npz = args.out_dir / f"{args.dataset}__p32tok_K{args.K}.npz"
    mjs = args.out_dir / f"{args.dataset}__p32tok_K{args.K}.meta.json"
    assert not npz.exists() and not mjs.exists(), f"{npz} exists"
    ds = load(args.dataset)
    splits = [sp for sp in X.SPLITS if sp in ds.texts]
    assert not ({"test", "oos_test"} & set(splits))
    aug_texts, aug_y, kept_sha = [], [], None
    if args.kept != "none":
        kp = Path(args.kept)
        kept_sha = WF.sha256_file(kp)[:16]
        for line in kp.read_text().splitlines():
            c = json.loads(line)
            if c["set"] != args.dataset:
                continue
            assert ds.texts["train"][c["index"]] == c["source"] and ds.intents[c["y"]] == c["label"], "kept row does not match train"
            aug_texts.append(c["text"])
            aug_y.append(int(c["y"]))
    assert args.table.startswith("wordtab:")
    TP = Path(args.table.split(":", 1)[1])
    zt = np.load(TP)
    live = np.abs(zt["codes"].astype(np.float32)).sum(1) > 0
    M = zt["codes"].astype(np.float32)
    words = [str(w) for w in zt["words"]]
    row = {w: i for i, w in enumerate(words) if live[i]}
    toks = {sp: [WF.glove_tokens(t) for t in ds.texts[sp]] for sp in splits}
    toks_aug = [WF.glove_tokens(t) for t in aug_texts]
    need = sorted({row[w] for sp in splits for x in toks[sp] for w in x if w in row} | {row[w] for x in toks_aug for w in x if w in row})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(np.asarray(M[need]), device=dev)
    ids = {sp: [[local[row[w]] for w in x if w in row] for x in toks[sp]] for sp in splits}
    ids_aug = [[local[row[w]] for w in x if w in row] for x in toks_aug]
    n_tok = {sp: sum(len(x) for x in toks[sp]) for sp in splits}
    n_in = {sp: sum(1 for x in toks[sp] for w in x if w in row) for sp in splits}
    name_of = lambda i: words[need[int(i)]]  # noqa: E731
    info = {"table": TP.stem, "table_sha256": WF.sha256_file(TP)[:16],
            "tokenizer": "lowercase, Stanford-style contraction split, regex " + WF.TOK.pattern,
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
    ref = np.load(args.ref_dir / f"{args.dataset}__p32tok_K{args.K}.npz")
    ref_meta = json.loads((args.ref_dir / f"{args.dataset}__p32tok_K{args.K}.meta.json").read_text())
    eq = {"ref_dir": str(args.ref_dir), "max_abs_diff": {k: float(np.abs(out[k] - ref[k]).max()) for k in sorted(out) if k in ref.files},
          "identical": all(np.array_equal(out[k], ref[k]) for k in sorted(out) if k in ref.files)
          and all(k in out for k in ref.files if k.split("__")[0] in splits),
          "concepts_identical": names == ref_meta["concepts"], "thresholds_identical": th == ref_meta["thresholds"]}
    n_orig = len(ds.texts["train"])
    if ids_aug:
        pa = X.pooled(ids_aug, E, En, Cn, dev)
        for k, v in pa.items():
            out[f"train__{k}"] = np.concatenate([out[f"train__{k}"], v])
    out["y_train"] = np.concatenate([ds.labels["train"], np.asarray(aug_y, np.int64)]).astype(ds.labels["train"].dtype)
    out["y_val"] = ds.labels["val"]
    out["n_orig"] = np.asarray(n_orig)
    meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": args.K,
            "members": np.bincount(a, minlength=args.K).tolist(), "dataset": args.dataset, **info,
            "wordtab_feats_sha256": "001a93b259a1b34f (path re-run by c3_feats.py)",
            "c3_feats_sha256": WF.sha256_file(Path(__file__).resolve())[:16], "arm": args.arm, "n_orig": n_orig,
            "n_aug": len(aug_texts), "kept": args.kept, "kept_sha256": kept_sha,
            "aug_sentences_without_token": int(sum(1 for x in ids_aug if not x)), "equivalence": eq}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **out)
    mjs.write_text(json.dumps(meta))
    print(json.dumps({"dataset": args.dataset, "arm": args.arm, "n_orig": n_orig, "n_aug": len(aug_texts),
                      "identical": eq["identical"], "concepts_identical": eq["concepts_identical"],
                      "thresholds_identical": eq["thresholds_identical"],
                      "max_abs_diff": max(eq["max_abs_diff"].values()), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
