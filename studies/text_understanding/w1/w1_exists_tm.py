#!/usr/bin/env python
"""W1 (VAL only): a copy of glove_exists_tm.py (the N7 GloVe arm = exists2_tm.py of test 2) with these changes and
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing: build it with w1_feats.py first"
    assert json.loads(cache.with_suffix(".meta.json").read_text()).get("table"), "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, "cuda")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ds = load(args.dataset)
    C = len(ds.intents)
    ytr, yva = F["y_train"], F["y_val"]
    view = X.View(args.view, meta).fit(F)
    names = [re.sub(r"^existsT\[(\d+)\]", lambda m: f"some word ~ {{{meta['concepts'][int(m.group(1))]}}}", n) for n in view.names]
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=2)
    tm = make_tm(C, Btr.shape[1], cfg, "half")
    tm.fit(Btr, ytr, Bva, yva)
    pv = tm.predict(torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")).cpu().numpy()
    hs = json.loads((EMB / "hardset.json").read_text())[args.dataset]
    rec = {"tag": args.tag, "posthoc": True, "table": meta.get("table", "potion-base-32M"), "table_sha256": meta.get("table_sha256"), "dataset": args.dataset, "view": args.view, "seed": args.seed,
           "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "cfg": asdict(cfg),
           "val_acc": round(tm.best_val, 4), "best_epoch": getattr(tm, "best_epoch", None),
           "train_s": round(tm.train_seconds, 1),
           "lost_fixed": round(float((pv[hs["lost"]] == yva[hs["lost"]]).mean()), 4),
           "won_kept": round(float((pv[hs["won"]] == yva[hs["won"]]).mean()), 4),
           "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats()}
    if "oos_val__mean" in F:
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
        Xoo = torch.as_tensor(view.transform(F, "oos_val"), dtype=torch.uint8, device="cuda")
        S = torch.cat([tm.class_sums(Xin), tm.class_sums(Xoo)]).cpu().numpy()
        is_oos = np.r_[np.zeros(len(Bva)), np.ones(len(F["oos_val__mean"]))]
        rec["oos_auroc_val_max_sum"] = round(X.auroc(is_oos, -np.sort(S, 1)[:, -1]), 4)
    vt = ds.texts["val"]
    rec["examples"] = {t: {"true": ds.intents[yva[vt.index(t)]], "tm": ds.intents[pv[vt.index(t)]]} for t in X.EXAMPLES if t in vt}
    rec["total_s"] = round(time.time() - t0, 1)
    with open(HERE / "results_w1_val.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    st = tm.state()
    classes = [ds.intents.index(c) for c in X.SHOW.get(args.dataset, []) if c in ds.intents]
    lines = top_rules(st["ta"], st["w"], names, ds.intents, n_states=cfg.n_states, per_class=2, classes=classes)
    for t in X.EXAMPLES[:1]:
        if t in vt:
            lines.append(f"receipt for {t!r}:\n" + decision_receipt(st["ta"], st["w"], Bva[vt.index(t)], names, ds.intents,
                                                                     n_states=cfg.n_states, top=2))
    with open(HERE / "rules_w1.txt", "a") as f:
        f.write(f"=== {args.tag} {args.dataset} {args.view} seed {args.seed} val {rec['val_acc']}\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats")}))


if __name__ == "__main__":
    main()
