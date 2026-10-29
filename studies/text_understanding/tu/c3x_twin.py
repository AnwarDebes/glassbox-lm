#!/usr/bin/env python
"""Step 3c follow-up (b), the count-only twin's texts (tu/PLAN_3c.md "Arms": "5 noisy copies of each original: random
deletion of each word with p 0.1, plus one random swap of two adjacent words (EDA's two lexicon-free operations; numpy
default_rng(20261022)), F1 only"). CPU only; it reads the TRAINING splits, and the evaluation texts only for F1.

    python -I -B c3x_twin.py --out tu/c3x/twin_kept.jsonl

- Words: the text split on whitespace. For each original, in the order CLINC150, Banking77, HWU64 and by training
  index, five copies from one numpy default_rng(20261022). Each copy drops each word with p 0.1 (when every word would
  go, the first word stays), then, when it has 2 or more words, swaps one random adjacent pair.
- F1, as c3_filter.py's: a copy is dropped when its key (build_corpus_c4.key) equals an evaluation text's or a MultiWOZ
  dev/test utterance's, when it shares an 8-gram with one (c3_filter.leak_index), or when its key equals a training
  original's key (same class: redundant; another class: a conflict).
- Output rows have kept_F.jsonl's fields (set, index, y, label, source, text); the counts per set and per reason are
  printed and written beside the output. Never overwrites.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")

SETS = ("clinc150", "banking77", "hwu64")
SEED, N_COPIES, P_DEL = 20261022, 5, 0.1


def eda_copy(words: list[str], rng: np.random.Generator) -> list[str]:
    keep = [w for w, r in zip(words, rng.random(len(words))) if r >= P_DEL]
    if not keep and words:
        keep = [words[0]]
    if len(keep) >= 2:
        j = int(rng.integers(0, len(keep) - 1))
        keep[j], keep[j + 1] = keep[j + 1], keep[j]
    return keep


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta_p = a.out.with_suffix(".meta.json")
    for p in (a.out, meta_p):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    import c3_filter as CF  # F1's leak index and 8-grams; imports build_corpus_c4, which hides the GPU (CPU job)
    from c3_common import B
    from dl.data import load

    t0 = time.time()
    keys, grams, leak = CF.leak_index()
    rng = np.random.default_rng(SEED)
    rows, counts = [], {}
    for name in SETS:
        ds = load(name)
        tr, ytr = ds.texts["train"], ds.labels["train"]
        tr_key: dict[str, set] = {}
        for t, y in zip(tr, ytr):
            tr_key.setdefault(B.key(t), set()).add(int(y))
        c = {"sources": len(tr), "copies": 0, "f1_eval_key": 0, "f1_eval_8gram": 0, "f1_train_same": 0, "f1_train_other": 0,
             "empty": 0, "kept": 0}
        for i, (t, y) in enumerate(zip(tr, ytr)):
            words = t.split()
            for _ in range(N_COPIES):
                cp = " ".join(eda_copy(words, rng))
                c["copies"] += 1
                if not B.norm(cp):
                    c["empty"] += 1
                    continue
                k = B.key(cp)
                if k in keys:
                    c["f1_eval_key"] += 1
                    continue
                if any(g in grams for g in CF.grams_of(cp)):
                    c["f1_eval_8gram"] += 1
                    continue
                if k in tr_key:
                    c["f1_train_same" if tr_key[k] == {int(y)} else "f1_train_other"] += 1
                    continue
                c["kept"] += 1
                rows.append({"set": name, "index": i, "y": int(y), "label": ds.intents[int(y)], "source": t, "text": cp})
        counts[name] = c
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"seed": SEED, "copies_per_original": N_COPIES, "p_delete": P_DEL, "counts": counts,
            "leak": {k: v for k, v in leak.items() if k != "files"}, "out_sha16": sha16(a.out),
            "c3x_twin.py": sha16(Path(__file__).resolve()), "c3_filter.py": sha16(HERE / "c3_filter.py"),
            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(meta_p, "x") as f:
        f.write(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("counts", "out_sha16", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
