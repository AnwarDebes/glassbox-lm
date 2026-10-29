#!/usr/bin/env python
"""W3 D0 (DESCRIPTIVE diagnostic, intent VAL only, no bar, no download; logged in LOG.md before the run): where does
potion beat the C4 counting table? CPU only.

For CLINC150, Banking77 and HWU64: the logistic-regression probe of W2 part 2 (dl.linear.fit_logreg, the C that
w2/results_w2_lr.jsonl chose per table and set, here on the CPU) on the existsT bits of potion (test 2's caches)
and of ppw_S1000 (W1-S's caches). Per validation example: which table's probe is right, and what its words look
like for the C4 table (glove_feats.glove_tokens; a token is out of vocabulary when ppw_S1000 has no row for it;
C4 counts from w1/corpus/c4_words_vocab.json). Groups: potion-only right, ppw-only right, both, neither.
If the potion-only group is rich in out-of-vocabulary or rare words, a lexicon of known words (W3) can do little
there; if its words are common, the gap is about meaning and a lexicon may help. The test split is never scored.
Writes w3/d0_errors.txt and one row to w3/results_w3.jsonl (never overwrites: a new name if the text exists).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from glove_feats import glove_tokens  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402

SETS = ("clinc150", "banking77", "hwu64")
DIRS = {"potion": EMB / "emb_potion_test2", "ppw_S1000": EMB / "w1" / "feats" / "ppw_S1000_d500_e0.5"}
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
VOCAB = EMB / "w1" / "corpus" / "c4_words_vocab.json"
RARE = 1000


def bits(name: str, ds: str):
    F, meta = X.features(ds, 512, DIRS[name], "cpu")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    v = X.View("existsT", meta).fit(F)
    return v.transform(F, "train"), v.transform(F, "val"), F["y_train"], F["y_val"]


def main() -> None:
    torch.set_num_threads(2)
    t0 = time.time()
    lr_rows = {}
    for x in (EMB / "w2" / "results_w2_lr.jsonl").read_text().splitlines():
        r = json.loads(x)
        lr_rows[(r["name"], r["dataset"])] = r
    z = np.load(TABLE)
    live = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
    known = {str(w) for w, l in zip(z["words"], live) if l}
    voc = json.loads(VOCAB.read_text())
    count = dict(zip(voc["words"], voc["counts"]))
    out, summary = [], {}
    for ds in SETS:
        pred, acc = {}, {}
        for name in DIRS:
            Btr, Bva, ytr, yva = bits(name, ds)
            C = lr_rows[(name, ds)]["C"]
            n_classes = int(max(ytr.max(), yva.max())) + 1
            W, b = fit_logreg(Btr.astype(np.float32), ytr, n_classes, C, device="cpu")
            pred[name] = logits(Bva.astype(np.float32), W, b, device="cpu").argmax(1).cpu().numpy()
            acc[name] = float((pred[name] == yva).mean())
            if name == "potion":
                y_ref = yva
            else:
                assert np.array_equal(yva, y_ref), "the two caches order val differently"
        data = load(ds)
        texts = data.texts["val"]
        assert np.array_equal(np.asarray(data.labels["val"]), y_ref) and len(texts) == len(y_ref)
        rp, rc = pred["potion"] == y_ref, pred["ppw_S1000"] == y_ref
        groups = {"potion_only": rp & ~rc, "ppw_only": rc & ~rp, "both": rp & rc, "neither": ~rp & ~rc}
        toks = [glove_tokens(t) for t in texts]
        oov = np.array([sum(w not in known for w in tk) / max(len(tk), 1) for tk in toks])
        any_oov = np.array([any(w not in known for w in tk) for tk in toks])
        rare = np.array([sum((w in known) and count.get(w, 0) < RARE for w in tk) / max(len(tk), 1) for tk in toks])
        ntok = np.array([len(tk) for tk in toks])
        s = {"n_val": int(len(texts)), "acc_cpu": acc,
             "acc_registered_gpu": {n: lr_rows[(n, ds)]["lr_val"] for n in DIRS},
             "C": {n: lr_rows[(n, ds)]["C"] for n in DIRS}}
        for g, m in groups.items():
            s[g] = {"n": int(m.sum()), "oov_token_share": round(float(oov[m].mean()), 4) if m.any() else None,
                    "share_with_an_oov_token": round(float(any_oov[m].mean()), 4) if m.any() else None,
                    "rare_token_share": round(float(rare[m].mean()), 4) if m.any() else None,
                    "mean_tokens": round(float(ntok[m].mean()), 2) if m.any() else None}
        s["all"] = {"oov_token_share": round(float(oov.mean()), 4), "share_with_an_oov_token": round(float(any_oov.mean()), 4),
                    "rare_token_share": round(float(rare.mean()), 4), "mean_tokens": round(float(ntok.mean()), 2)}
        po = np.flatnonzero(groups["potion_only"])
        oov_words = Counter(w for i in po for w in toks[i] if w not in known)
        s["potion_only_top_oov_words"] = oov_words.most_common(15)
        summary[ds] = s
        out.append(f"== {ds}: val {len(texts)}; LR (CPU) potion {acc['potion']:.4f} / ppw_S1000 {acc['ppw_S1000']:.4f} "
                   f"(registered GPU rows {s['acc_registered_gpu']['potion']} / {s['acc_registered_gpu']['ppw_S1000']})")
        for g in ("potion_only", "ppw_only", "both", "neither", "all"):
            e = s[g]
            out.append(f"  {g:12s} n {e.get('n', len(texts)):5d}  OOV token share {e['oov_token_share']}  "
                       f"with an OOV token {e['share_with_an_oov_token']}  rare (<{RARE} in C4) {e['rare_token_share']}  "
                       f"tokens {e['mean_tokens']}")
        out.append(f"  most frequent OOV words in potion-only: {s['potion_only_top_oov_words']}")
        names = data.intents
        out.append("  first 25 potion-only examples ([w] = no C4 row): text | true | ppw_S1000 said")
        for i in po[:25]:
            shown = " ".join(f"[{w}]" if w not in known else w for w in toks[i])
            out.append(f"    {shown} | {names[y_ref[i]]} | {names[pred['ppw_S1000'][i]]}")
    txt = "\n".join(out)
    print(txt)
    p = HERE / "d0_errors.txt"
    if p.exists():
        p = HERE / f"d0_errors_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.txt"
    p.write_text(txt + "\n")
    with open(HERE / "results_w3.jsonl", "a") as f:
        f.write(json.dumps({"study": "W3 D0 (descriptive, val)", "summary": summary, "seconds": round(time.time() - t0, 1),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


if __name__ == "__main__":
    main()
