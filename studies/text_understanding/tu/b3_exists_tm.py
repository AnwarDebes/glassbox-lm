#!/usr/bin/env python
"""Text understanding, step 3b (VAL only; tu/PLAN_3b.md "Arms and readers"): tu/c3_exists_tm.py (6474ebb07393f433) with these
changes and nothing else: (1) --block NPZ appends a bit block after the view's bits (keys "train", "val" and, when the
set has it, "oos_val": np.packbits rows in the cache's row order, and "n_features"); a block without "oos_val" skips the
OOS line; (2) --arm names the arm (default "base"); (3) rows go to tu/b3/results_b3_val.jsonl and rules to
tu/b3/rules_b3.txt, and record "study": "3b", the arm and the block's path, sha256 and width. With no block it equals
c3_exists_tm.py (the code check). Docstring of c3_exists_tm.py:

Text understanding, step 3c (VAL only; tu/PLAN_3c.md "Features and readers"): tu/b0_exists_tm.py (9fbc334af9aea8ff)
with these changes and nothing else: (1) the existsT thermometer is fit on the first n_orig training rows only (the
originals; the feature cache's "n_orig", or every row when it has none), so the bit definition stays the
reference's and the extra (paraphrase) rows only add examples; (2) rows go to tu/c3/results_c3_val.jsonl and rules to
tu/c3/rules_c3.txt, and record "study": "3c", the arm (the cache meta's "arm", or "orig") and n_orig / n_train.
With no extra rows it equals b0_exists_tm.py. Docstring of b0_exists_tm.py:

Text understanding, step 0b (VAL only; PLAN_text_understanding.md section 1b): w3/w3_exists_tm.py (25b2a0c385e8b455)
with these changes and nothing else: (1) --s sets the TM's s (default 8.0, the harness value); (2) rows go to
tu/results_0b_val.jsonl and rules to tu/rules_0b.txt (this copy lives in tu/, so HERE is tu and EMB is still
glassbox-lm/studies/text_understanding); (3) the potion feature cache, whose meta names no table, is accepted by its exact folder
(emb_potion_test2, test 2's cache) and recorded as potion-base-32M; any other cache must still name its
table; (4) each row records "study": "0b", the s and the feature folder. Docstring of w3_exists_tm.py:

W3 step 1 (VAL only): w2/w2_exists_tm.py (e326b2a2ed8252b7) with one change and nothing else: rows go to
w3/results_w3_val.jsonl and rules to w3/rules_w3.txt (this copy lives in w3/, so HERE is w3 and EMB is still
glassbox-lm/studies/text_understanding). The "posthoc": true flag of the rows is inherited (it marks that test 2 chose the existsT
view post hoc); W3's runs are exploratory with rules fixed before them. Docstring of w2_exists_tm.py:

W2 H-B part 2 (VAL only): w1/w1_exists_tm.py with one change and nothing else: rows go to w2/results_w2_val.jsonl and
rules to w2/rules_w2.txt (this copy lives in w2/, so HERE is w2 and EMB is still glassbox-lm/studies/text_understanding). Docstring of
w1_exists_tm.py:

W1 (VAL only): a copy of glove_exists_tm.py (the N7 GloVe arm = exists2_tm.py of test 2) with these changes and
nothing else: rows go to w1/results_w1_val.jsonl and rules to w1/rules_w1.txt, each row records the table and its
sha256 from the feature cache meta (w1_feats.py), hardset.json is read from glassbox-lm/studies/text_understanding, and the run stops
unless the feature cache already exists and its meta names a table (exists_tm.features would otherwise build
POTION features into an empty folder, so a missing W1 cache would silently become a potion row). The features,
thermometer, TM settings and seeds are exists2_tm.py's.

Original docstring of exists2_tm.py:
POST HOC (chosen after the seed-1 rows of test 2 and posthoc_lr.jsonl, VAL only).

    exists2_tm.py --dataset clinc150 --view existsT|mean+existsT --seed 1 --emb-dir DIR

existsT = the same existential similarities as exists_tm.py (max over the sentence's tokens of the cosine to
each of the 512 concepts), binarized with the thermometer (per-concept quantiles over the training sentences,
7 thresholds) instead of the member percentiles. A bit is still "some word has cos >= t to concept k" with t a
constant per concept, so existsT alone is still a lookup and a bitwise OR at run time. Reason: with member
thresholds only 1% of the bits are on and LR on them loses (CLINC150 0.912), while LR on thermometer bits gets
0.9473 alone and 0.956 with the mean (posthoc_lr.jsonl). Everything else is exists_tm.py, unchanged.
Appends to results_posthoc.jsonl and rules_posthoc.txt.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from dl.data import load  # noqa: E402
from dl.explain import decision_receipt, top_rules  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def X_sha16(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--block", type=Path, default=None)
    ap.add_argument("--arm", default="base")
    args = ap.parse_args()

    t0 = time.time()
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing: build it with w1_feats.py first"
    potion_dir = (EMB / "emb_potion_test2").resolve()
    is_potion = args.emb_dir.resolve() == potion_dir
    assert is_potion or json.loads(cache.with_suffix(".meta.json").read_text()).get("table"), "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, "cuda")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ds = load(args.dataset)
    C = len(ds.intents)
    ytr, yva = F["y_train"], F["y_val"]
    n_orig = int(F["n_orig"]) if "n_orig" in F else len(F["y_train"])  # 3c: the originals come first in "train"
    view = X.View(args.view, meta).fit({k: (v[:n_orig] if k.startswith("train__") else v) for k, v in F.items()})
    names = [re.sub(r"^existsT\[(\d+)\]", lambda m: f"some word ~ {{{meta['concepts'][int(m.group(1))]}}}", n) for n in view.names]
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    blk = np.load(args.block) if args.block is not None else None
    nb = int(blk["n_features"]) if blk is not None else 0
    if blk is not None:
        bt, bv = (np.unpackbits(blk[k], axis=1)[:, :nb] for k in ("train", "val"))
        assert bt.shape[0] == Btr.shape[0] and bv.shape[0] == Bva.shape[0], (bt.shape, Btr.shape, bv.shape, Bva.shape)
        Btr, Bva = np.concatenate([Btr, bt], 1), np.concatenate([Bva, bv], 1)
        names = names + [f"block[{j}]" for j in range(nb)]
    cfg = TMConfig(n_clauses=4000, T=4000, s=args.s, epochs=40, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=2)
    tm = make_tm(C, Btr.shape[1], cfg, "half")
    tm.fit(Btr, ytr, Bva, yva)
    pv = tm.predict(torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")).cpu().numpy()
    hs = json.loads((EMB / "hardset.json").read_text())[args.dataset]
    rec = {"tag": args.tag, "study": "3b", "arm": args.arm, "block": str(args.block) if blk is not None else None,
           "block_sha256": X_sha16(args.block) if blk is not None else None, "block_bits": nb, "cache_arm": meta.get("arm", "orig"), "n_orig": n_orig, "n_train": int(len(ytr)), "s": args.s, "emb_dir": str(args.emb_dir), "posthoc": True, "table": meta.get("table") or ("potion-base-32M" if is_potion else None), "table_sha256": meta.get("table_sha256"), "dataset": args.dataset, "view": args.view, "seed": args.seed,
           "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "cfg": asdict(cfg),
           "val_acc": round(tm.best_val, 4), "best_epoch": getattr(tm, "best_epoch", None),
           "train_s": round(tm.train_seconds, 1),
           "lost_fixed": round(float((pv[hs["lost"]] == yva[hs["lost"]]).mean()), 4),
           "won_kept": round(float((pv[hs["won"]] == yva[hs["won"]]).mean()), 4),
           "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats()}
    if "oos_val__mean" in F and (blk is None or "oos_val" in blk.files):
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
        Boo = view.transform(F, "oos_val")
        if blk is not None:
            Boo = np.concatenate([Boo, np.unpackbits(blk["oos_val"], axis=1)[:, :nb]], 1)
        Xoo = torch.as_tensor(Boo, dtype=torch.uint8, device="cuda")
        S = torch.cat([tm.class_sums(Xin), tm.class_sums(Xoo)]).cpu().numpy()
        is_oos = np.r_[np.zeros(len(Bva)), np.ones(len(F["oos_val__mean"]))]
        rec["oos_auroc_val_max_sum"] = round(X.auroc(is_oos, -np.sort(S, 1)[:, -1]), 4)
    vt = ds.texts["val"]
    rec["examples"] = {t: {"true": ds.intents[yva[vt.index(t)]], "tm": ds.intents[pv[vt.index(t)]]} for t in X.EXAMPLES if t in vt}
    rec["total_s"] = round(time.time() - t0, 1)
    with open(HERE / "b3" / "results_b3_val.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    st = tm.state()
    classes = [ds.intents.index(c) for c in X.SHOW.get(args.dataset, []) if c in ds.intents]
    lines = top_rules(st["ta"], st["w"], names, ds.intents, n_states=cfg.n_states, per_class=2, classes=classes)
    for t in X.EXAMPLES[:1]:
        if t in vt:
            lines.append(f"receipt for {t!r}:\n" + decision_receipt(st["ta"], st["w"], Bva[vt.index(t)], names, ds.intents,
                                                                     n_states=cfg.n_states, top=2))
    with open(HERE / "b3" / "rules_b3.txt", "a") as f:
        f.write(f"=== {args.tag} {rec['table']} {args.dataset} {args.view} s {args.s} seed {args.seed} val {rec['val_acc']}\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats")}))


if __name__ == "__main__":
    main()
