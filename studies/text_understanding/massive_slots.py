#!/usr/bin/env python
"""Test 4 data (LOG.md): MASSIVE en-US slot filling, one example per word.

    massive_slots.py --emb-dir DIR

Reads the decision-layer study's MASSIVE file (read only). Each utterance's annot_utt ("wake me up at [time : five
am]") gives one slot label per word ("Other" outside the brackets); the words equal utt.split() for every row
(checked). Each word is tokenized alone with the potion-base-32M tokenizer, and its meaning bits are the OR of its
tokens' node bits from graph_feats.py (massive_en__graph_K512.npz: "cos(token, concept k) >= t", the existsT
thresholds of that dataset). Writes DIR/massive_en__slots_K512.npz: per split the word bits (packed), the word's
index in the training-word vocabulary (-1 if unseen), the label index, the utterance offsets; plus label and word
names. Scoring is in slot_eval.py.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exists_tm as X  # noqa: E402  (HF_ID, REV of the static table)

SRC = Path(f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/massive/en-US.jsonl")
PART = {"train": "train", "val": "dev", "test": "test"}
PAT = re.compile(r"\[([^\]:]+?) : ([^\]]*?)\]")


def words_labels(annot: str):
    words, labs, pos = [], [], 0
    for m in PAT.finditer(annot):
        pre = annot[pos : m.start()].split()
        span = m.group(2).split()
        words += pre + span
        labs += ["Other"] * len(pre) + [m.group(1).strip()] * len(span)
        pos = m.end()
    tail = annot[pos:].split()
    return words + tail, labs + ["Other"] * len(tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, required=True)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    tok = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)[0].tokenizer
    g = np.load(args.emb_dir / "massive_en__graph_K512.npz")
    n_sym = int(g["n_sym"])
    used = g["used"]
    node = np.unpackbits(g["node_bits"], axis=1)[:, :n_sym].astype(bool)
    row = {int(t): i for i, t in enumerate(used)}
    rows = [json.loads(x) for x in SRC.read_text().splitlines() if x.strip()]
    data = {sp: [] for sp in PART}
    for r in rows:
        w, lab = words_labels(r["annot_utt"])
        assert w == r["utt"].split(), r["id"]
        for sp, p in PART.items():
            if r["partition"] == p:
                data[sp].append((w, lab))
    train_labels = sorted({l for w, ls in data["train"] for l in ls} - {"Other"})
    extra = sorted({l for sp in ("val", "test") for w, ls in data[sp] for l in ls} - set(train_labels) - {"Other"})
    labels = ["Other"] + train_labels + extra  # labels seen only in val / test come last: no model can predict them
    lid = {l: i for i, l in enumerate(labels)}
    vocab = sorted({x for w, _ in data["train"] for x in w})
    wid = {x: i for i, x in enumerate(vocab)}
    out, missing, total = {}, 0, 0
    for sp, items in data.items():
        words = [x for w, _ in items for x in w]
        ids = [e.ids for e in tok.encode_batch(words, add_special_tokens=False)]
        bits = np.zeros((len(words), n_sym), bool)
        for i, t in enumerate(ids):
            for k in t:
                total += 1
                if k in row:
                    bits[i] |= node[row[k]]
                else:
                    missing += 1
        out[f"{sp}__bits"] = np.packbits(bits, axis=1)
        out[f"{sp}__wid"] = np.array([wid.get(x, -1) for x in words], np.int64)
        out[f"{sp}__y"] = np.array([lid[l] for _, ls in items for l in ls], np.int64)
        out[f"{sp}__off"] = np.cumsum([0] + [len(w) for w, _ in items]).astype(np.int64)
    out["n_sym"] = np.array(n_sym)
    path = args.emb_dir / "massive_en__slots_K512.npz"
    np.savez(path, **out)
    meta = {"labels": labels, "vocab": vocab, "missing_token_bits": missing, "tokens": total,
            "n_train_labels": 1 + len(train_labels), "eval_only_labels": extra,
            "eval_words_with_eval_only_labels": {sp: int((out[f"{sp}__y"] > len(train_labels)).sum()) for sp in ("val", "test")},
            "words": {sp: int(len(out[f"{sp}__y"])) for sp in PART}, "utterances": {sp: len(v) for sp, v in data.items()},
            "other_share_train": round(float((out["train__y"] == 0).mean()), 3)}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("labels", "vocab")}), len(labels), "labels")


if __name__ == "__main__":
    main()
