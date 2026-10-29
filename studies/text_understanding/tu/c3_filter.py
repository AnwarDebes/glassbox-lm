#!/usr/bin/env python
"""Step 3c, the filters (tu/PLAN_3c.md "Filters"): F1 leakage and conflicts, F1b near-copies of evaluation texts, F2
teacher cosine, F3 label consistency.

    python -I c3_filter.py --gen tu/c3/gen.jsonl --out-dir tu/c3 --device cuda          # the GPU job
    python -I c3_filter.py --gen tu/c3/pilot/gen_pilot.jsonl --out-dir tu/c3/pilot --device cpu --pilot

F1: a paraphrase is dropped when its key (B.key) equals, or it shares an 8-gram of normalised words (B.norm, B.NGRAM)
with, any text of B.eval_texts() (every evaluation split: the intent sets' val and test, Banking77's carved val,
MASSIVE, TREC, SST-2, MR, R8, R52, ABCD and STAR dev and test) or any MultiWOZ 2.2 dev or test utterance (read as
a3_mwoz_filter.py reads them); or when its key equals a training original's key of the same set (another class: a
conflict; the same class: redundant).
F1b (the fix, 22:21Z): the 8-gram rule cannot see a near-copy of a short utterance, so a paraphrase is also
dropped when its bge cosine to some val or test text of its own set (CLINC150: also oos_val and oos_test) is at least
0.98. It reads those texts only, never their labels. Reported per set: the drops, their share of the F1 survivors, and
the natural share of training originals at or above 0.98 to the same texts.
F2: bge-base-en-v1.5 (a3_teacher.py's load and encode: [CLS], L2-normalised, at most 128 tokens; fp16 on cuda, fp32
on cpu) cosine to its source at least 0.7.
F3: a 5-fold cross-fitted multinomial logistic regression (dl.linear.fit_logreg, C 10) on the bge embeddings of the
set's training originals; folds from a numpy default_rng(20261021) permutation of the source indices (position j -> fold
j mod 5); a paraphrase is kept when the fold model that did not see its source predicts the source's label.
Writes OUT/kept_F.jsonl (F1+F1b+F2+F3, the registered arm), OUT/kept_C.jsonl (F1+F1b+F2, descriptive) and
OUT/filter_meta.json; with --pilot also OUT/pilot_examples.tsv. Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import glob
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import a3_teacher as TEA  # noqa: E402
import c3_common as C  # noqa: E402
from c3_common import B  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402

MW = Path(f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/multiwoz22")
COS_MIN, NEAR_MAX, C_LR, FOLDS = 0.7, 0.98, 10.0, 5
EVAL_SPLITS = ("val", "test", "oos_val", "oos_test")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def grams_of(t: str) -> list[str]:
    w = B.norm(t).split()
    return [" ".join(w[j : j + B.NGRAM]) for j in range(len(w) - B.NGRAM + 1)]


def max_cos(a: np.ndarray, b: np.ndarray, device: str) -> np.ndarray:
    out = np.zeros(len(a), np.float32)
    if len(a) == 0 or len(b) == 0:
        return out
    Bm = torch.as_tensor(b.astype(np.float32), device=device)
    for s in range(0, len(a), 4096):
        S = torch.as_tensor(a[s : s + 4096].astype(np.float32), device=device) @ Bm.T
        out[s : s + 4096] = S.max(1).values.cpu().numpy()
    return out


def leak_index():
    ev, files = B.eval_texts()
    texts = [t for v in ev.values() for t in v]
    for split in ("dev", "test"):
        for f in sorted(glob.glob(str(MW / split / "dialogues_*.json"))):
            files[f"multiwoz22/{split}/{Path(f).name}"] = sha16(Path(f))
            for dlg in json.loads(Path(f).read_text()):
                texts += [turn["utterance"] for turn in dlg["turns"]]
    keys = {B.key(t) for t in texts if B.norm(t)}
    grams = {g for t in texts for g in grams_of(t)}
    return keys, grams, {"eval_texts": len(texts), "keys": len(keys), "eight_grams": len(grams), "files": files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    outs = {"F": args.out_dir / "kept_F.jsonl", "C": args.out_dir / "kept_C.jsonl", "meta": args.out_dir / "filter_meta.json",
            "ex": args.out_dir / "pilot_examples.tsv"}
    for p in outs.values():
        assert not p.exists(), f"{p} exists"
    rows = [json.loads(l) for l in args.gen.read_text().splitlines()]
    keys, grams, leak_meta = leak_index()
    TEA.verify_files()
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok, model = TEA.load(args.device, dtype)
    meta = {"gen": str(args.gen), "gen_sha256": sha16(args.gen), "leak": {k: v for k, v in leak_meta.items() if k != "files"},
            "leak_files": leak_meta["files"], "device": args.device, "near_max": NEAR_MAX, "cos_min": COS_MIN, "sets": {}}
    kept_F, kept_C, examples = [], [], []
    for name in C.SETS:
        rs = [r for r in rows if r["set"] == name]
        if not rs:
            continue
        ds = load(name)
        tr, ytr = ds.texts["train"], ds.labels["train"]
        assert all(tr[r["index"]] == r["text"] and ds.intents[int(ytr[r["index"]])] == r["label"] for r in rs), "row does not match train"
        tr_key = {}
        for t, y in zip(tr, ytr):
            tr_key.setdefault(B.key(t), set()).add(int(y))
        cand, cnt = [], {"sources": len(rs), "paraphrases": 0, "f1_eval_key": 0, "f1_eval_8gram": 0, "f1_train_same": 0,
                          "f1_train_other": 0, "f1b_nearcopy": 0, "f2_cos": 0, "f3_label": 0, "kept_C": 0, "kept_F": 0,
                          "sources_lt5": 0}
        for r in rs:
            y = ds.intents.index(r["label"])
            cnt["sources_lt5"] += len(r["paraphrases"]) < C.N_PARA
            for p in r["paraphrases"]:
                cnt["paraphrases"] += 1
                k = B.key(p)
                why = None
                if k in keys:
                    why = "f1_eval_key"
                elif any(g in grams for g in grams_of(p)):
                    why = "f1_eval_8gram"
                elif k in tr_key:
                    why = "f1_train_same" if tr_key[k] == {y} else "f1_train_other"
                if why:
                    cnt[why] += 1
                    examples.append((name, r["label"], r["text"], p, why, ""))
                    continue
                cand.append({"set": name, "index": r["index"], "y": y, "label": r["label"], "source": r["text"], "text": p})
        # F1b: near-copies of this set's val and test texts (texts only)
        ev_texts = [t for sp in EVAL_SPLITS if sp in ds.texts for t in ds.texts[sp]]
        ev, _ = TEA.encode(tok, model, ev_texts, args.device, batch=256)
        etr, tr_info = TEA.encode(tok, model, tr, args.device, batch=256)
        orig_near = max_cos(etr, ev, args.device)
        cos = np.zeros(len(cand), np.float32)
        near = np.zeros(len(cand), np.float32)
        ep = np.zeros((0, etr.shape[1]), np.float16)
        if cand:
            ep, _ = TEA.encode(tok, model, [c["text"] for c in cand], args.device, batch=256)
            es, _ = TEA.encode(tok, model, [c["source"] for c in cand], args.device, batch=256)
            cos = (ep.astype(np.float32) * es.astype(np.float32)).sum(1)
            near = max_cos(ep, ev, args.device)
        # F3: cross-fitted LR on the bge embeddings of all training originals of the set
        perm = np.random.default_rng(20261021).permutation(len(tr))
        fold = np.empty(len(tr), np.int64)
        fold[perm] = np.arange(len(tr)) % FOLDS
        pred = np.full(len(cand), -1)
        cf = np.array([fold[c["index"]] for c in cand]) if cand else np.zeros(0, np.int64)
        for k in range(FOLDS):
            m = fold != k
            W, b = fit_logreg(etr[m].astype(np.float32), ytr[m], len(ds.intents), C_LR, device=args.device)
            sel = np.where(cf == k)[0]
            assert not np.isin([cand[i]["index"] for i in sel], np.where(m)[0]).any(), "a source was seen by its judge"
            if len(sel):
                pred[sel] = logits(ep[sel].astype(np.float32), W, b, device=args.device).argmax(1).cpu().numpy()
        for i, c in enumerate(cand):
            c["cos"] = round(float(cos[i]), 4)
            c["near_eval"] = round(float(near[i]), 4)
            c["f3_pred"] = ds.intents[int(pred[i])]
            if near[i] >= NEAR_MAX:
                cnt["f1b_nearcopy"] += 1
                examples.append((name, c["label"], c["source"], c["text"], "f1b_nearcopy", f"{near[i]:.3f}"))
                continue
            if c["cos"] < COS_MIN:
                cnt["f2_cos"] += 1
                examples.append((name, c["label"], c["source"], c["text"], "f2_cos", f"{c['cos']:.3f}"))
                continue
            cnt["kept_C"] += 1
            kept_C.append(c)
            if int(pred[i]) != c["y"]:
                cnt["f3_label"] += 1
                examples.append((name, c["label"], c["source"], c["text"], f"f3_label->{c['f3_pred']}", f"{c['cos']:.3f}"))
                continue
            cnt["kept_F"] += 1
            kept_F.append(c)
            examples.append((name, c["label"], c["source"], c["text"], "kept", f"{c['cos']:.3f}"))
        cnt["f1_survivors"] = len(cand)
        cnt["f1b_share_of_f1_survivors"] = round(cnt["f1b_nearcopy"] / max(1, len(cand)), 4)
        cnt["originals_nearcopy_share"] = round(float((orig_near >= NEAR_MAX).mean()), 4)
        cnt["f1b_eval_texts"] = len(ev_texts)
        cnt["mean_cos_candidates"] = round(float(cos.mean()), 4) if cand else None
        cnt["train_encode"] = tr_info
        meta["sets"][name] = cnt
        print(json.dumps({name: cnt}), flush=True)
    for key, rowsk in (("F", kept_F), ("C", kept_C)):
        outs[key].write_text("".join(json.dumps(c) + "\n" for c in rowsk))
        meta[f"kept_{key}_sha256"] = sha16(outs[key])
    meta.update({"c3_filter.py": sha16(Path(__file__).resolve()), "c3_common.py": sha16(HERE / "c3_common.py"),
                 "a3_teacher.py": sha16(HERE / "a3_teacher.py"), "seconds": round(time.time() - t0, 1),
                 "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    outs["meta"].write_text(json.dumps(meta, indent=1))
    if args.pilot:
        outs["ex"].write_text("set\tlabel\tsource\tparaphrase\toutcome\tvalue\n" + "".join("\t".join(e) + "\n" for e in examples))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("leak_files", "sets")}), flush=True)


if __name__ == "__main__":
    main()
