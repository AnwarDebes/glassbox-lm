#!/usr/bin/env python
"""M2, the registered test on MTOP English (tu/PLAN_M2.md "Features"): the features of one table on MTOP English,
from ONE k-means, with the TEST split included.

    python -I -B m2_feats.py --dataset mtop_en --table wordtab:TABLE.npz --arms a3B:none --out-root tu/m2/feats
    python -I -B m2_feats.py --dataset mtop_en --table potion --arms potion:none --out-root tu/m2/feats

tu/m1_feats.py (bb93c24f72edfe67) with these changes and nothing else (dataset changes only):
(1) the dataset comes from m2_data.load (MTOP English from the pinned zip: train = en/train, val = en/eval, test =
    en/test; the classes are en/train's top-level intents; an eval or test intent not in en/train has label -1) in
    place of dl.data.load; --load FILE.py:function (tests, the code check and the dry runs only) still takes a
    stand-in (m2_standin.py:load for the CLINC150 code check, m2_data.py:load_standin or m2_standin.py:mtop_small for
    MTOP), loaded under the module name "m2_hook";
(2) the meta's code-hash key is "m2_feats_sha256", and the meta records the loader's unseen-intent counts
    ("unseen_intents", null for a stand-in).
Unchanged from M1: the splits train, val, test (and oos_train, oos_val where a set has them; never oos_test); "test"
is pooled with the same table, concepts and thresholds as the other splits, and y_test is written; --table potion
builds the potion-base-32M features by exists_tm.features' path (the HF snapshot exists_tm.HF_ID at exists_tm.REV,
read with HF_HUB_OFFLINE=1); the k-means centroids are saved beside each arm's cache; the k-means and the thresholds
are fit on the ORIGINAL training tokens only. The arms with paraphrases (kept files) still work, but M2 uses none.
Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before any Hugging Face import: the potion snapshot is read, never fetched

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(EMB / "w1"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X
import m2_data
import wordtab_feats as WF
from dl.boolean import _spherical_kmeans

SPLITS = ("train", "val", "test", "oos_train", "oos_val")  # never oos_test


def load_ds(name: str, hook: str = ""):
    """m2_data.load (MTOP English), or (tests, code check, dry runs) FILE.py:function, a stand-in dataset whose "test"
    is not the real test split."""
    if not hook:
        return m2_data.load(name)
    import importlib.util

    path, fn = hook.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("m2_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)(name)


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


class WordTable:
    """A word table in w1's format: wordtab_feats.glove_tokens, then the table's live rows."""

    def __init__(self, path: Path):
        zt = np.load(path)
        live = np.abs(zt["codes"].astype(np.float32)).sum(1) > 0
        self.M = zt["codes"].astype(np.float32)
        self.words = [str(w) for w in zt["words"]]
        self.row = {w: i for i, w in enumerate(self.words) if live[i]}
        self.name, self.sha = path.stem, WF.sha256_file(path)[:16]

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [[self.row[w] for w in WF.glove_tokens(t) if w in self.row] for t in texts]

    def token(self, i: int) -> str:
        return self.words[i]


class Potion:
    """potion-base-32M by exists_tm.features' path: its tokenizer (no special tokens) and its static embedding rows."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
        self.tok = st[0].tokenizer
        self.M = st[0].embedding.weight.detach().float().numpy()
        self.name, self.sha = "potion-base-32M", f"{X.HF_ID}@{X.REV}"

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def token(self, i: int) -> str:
        return self.tok.id_to_token(i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True, help="wordtab:PATH (a word table npz in w1's format) or potion")
    ap.add_argument("--arms", required=True, help="arm:kept_file pairs, comma-separated; kept_file 'none' = originals only")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--device", default="cuda", help="cpu only for a functional test")
    ap.add_argument("--load", default="", help="tests, code check, dry runs: FILE.py:function returning a stand-in dataset")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    arms = [tuple(x.split(":", 1)) for x in args.arms.split(",")]
    outs = {arm: (args.out_root / arm / f"{args.dataset}__p32tok_K{args.K}.npz",
                  args.out_root / arm / f"{args.dataset}__p32tok_K{args.K}.meta.json",
                  args.out_root / arm / f"{args.dataset}__p32tok_K{args.K}.centroids.npy") for arm, _ in arms}
    for paths in outs.values():
        for p in paths:
            assert not p.exists(), f"{p} exists"
    ds = load_ds(args.dataset, args.load)
    splits = [sp for sp in SPLITS if sp in ds.texts]
    assert "test" in splits and "oos_test" not in splits
    aug = {arm: read_kept(path, ds, args.dataset) for arm, path in arms}
    if args.table == "potion":
        assert all(not aug[arm][0] for arm, _ in arms), "the potion arm has no paraphrases"
        tab = Potion()
    else:
        assert args.table.startswith("wordtab:")
        tab = WordTable(Path(args.table.split(":", 1)[1]))
    gids = {sp: tab.ids(ds.texts[sp]) for sp in splits}
    gids_aug = {arm: tab.ids(aug[arm][0]) for arm, _ in arms}
    need = sorted({g for sp in splits for x in gids[sp] for g in x} | {g for arm, _ in arms for x in gids_aug[arm] for g in x})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(np.asarray(tab.M[need]), device=dev)
    ids = {sp: [[local[g] for g in x] for x in gids[sp]] for sp in splits}
    name_of = lambda i: tab.token(need[int(i)])
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
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
    base = {}
    for sp, x in ids.items():
        for k, v in X.pooled(x, E, En, Cn, dev).items():
            base[f"{sp}__{k}"] = v
    n_orig = len(ds.texts["train"])
    report = {}
    for arm, path in arms:
        texts, ys, ksha = aug[arm]
        out = dict(base)
        ids_aug = [[local[g] for g in x] for x in gids_aug[arm]]
        if ids_aug:
            pa = X.pooled(ids_aug, E, En, Cn, dev)
            for k, v in pa.items():
                out[f"train__{k}"] = np.concatenate([base[f"train__{k}"], v])
        out["y_train"] = np.concatenate([ds.labels["train"], np.asarray(ys, np.int64)]).astype(ds.labels["train"].dtype)
        out["y_val"] = ds.labels["val"]
        out["y_test"] = ds.labels["test"]
        out["n_orig"] = np.asarray(n_orig)
        meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": args.K,
                "members": np.bincount(a, minlength=args.K).tolist(), "dataset": args.dataset, "table": tab.name,
                "table_sha256": tab.sha, "m2_feats_sha256": WF.sha256_file(Path(__file__).resolve())[:16],
                "arm": arm, "n_orig": n_orig, "n_aug": len(texts), "kept": path, "kept_sha256": ksha, "splits": splits,
                "kmeans_run_id": run_id, "concepts_sha16": hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16],
                "thresholds_sha16": hashlib.sha256(json.dumps(th).encode()).hexdigest()[:16], "arms_in_this_run": [x for x, _ in arms],
                "aug_sentences_without_token": int(sum(1 for x in ids_aug if not x)),
                "test_sentences_without_token": int(sum(1 for x in ids["test"] if not x)),
                "unseen_intents": ds.sources.get("unseen_intents")}
        npz, mjs, cnp = outs[arm]
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **out)
        np.save(cnp, cents)
        mjs.write_text(json.dumps(meta))
        report[arm] = {"n_aug": len(texts), "npz_sha16": WF.sha256_file(npz)[:16]}
    print(json.dumps({"dataset": args.dataset, "table": tab.name, "kmeans_run_id": run_id, "arms": report,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
