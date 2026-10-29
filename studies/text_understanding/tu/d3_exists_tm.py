#!/usr/bin/env python
"""Step 3d (VAL only; tu/PLAN_3d.md "Readers"): the TM reader of one arm, set and seed.

    python -I -B d3_exists_tm.py --dataset clinc150 --view existsT --seed 7 --s 8 --emb-dir tu/d3/feats/J --tag d3

tu/m1_tm.py (06ccf8eba7505de7) with these changes and nothing else:
(1) no test split: the cache (tu/d3_feats.py) has none, nothing is predicted beyond val, and no predictions file is
    written;
(2) rows go to <out-dir>/results_d3_val.jsonl (default tu/d3) and rules to <out-dir>/rules_d3.txt, and record
    "study": "3d", the UTC time, and the sha256 of this file, exists_tm.py and the cache;
(3) <out-dir> is created when it is missing (the parent created it only through its test-predictions folder);
(4) each row also records "key" (the run's name), "n_val" and "val_correct", the count of correct val predictions at
    the restored best-val state, which must equal the best val accuracy times n_val (else the run stops); the val
    predictions go to <out-dir>/preds_val/<key>.npy, written exclusively (one file per run).
--device and --smoke exist for the CPU tests only (--smoke: 200 clauses, T 200, 2 epochs, and its rows say so); --load
takes a stand-in dataset for the tests (m1_standin.py). The thermometer is fit on the first n_orig training rows.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X
from dl.explain import decision_receipt, top_rules
from dl.tm import TMConfig
from dl.tm_init import make_tm

from m1_feats import load_ds


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "d3")
    ap.add_argument("--device", default="cuda", help="cpu only for the tests")
    ap.add_argument("--smoke", action="store_true", help="tests only: 200 clauses, T 200, 2 epochs")
    ap.add_argument("--load", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    args = ap.parse_args()

    t0 = time.time()
    dev = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing: build it with d3_feats.py first"
    assert json.loads(cache.with_suffix(".meta.json").read_text()).get("table"), "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, dev)
    assert "test__exists" not in F and "y_test" not in F, "a 3d cache has no test rows"
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ds = load_ds(args.dataset, args.load)
    C = len(ds.intents)
    ytr, yva = F["y_train"], F["y_val"]
    assert (yva == ds.labels["val"]).all()
    n_orig = int(F["n_orig"]) if "n_orig" in F else len(F["y_train"])  # the originals come first in "train"
    if args.repeat > 1:
        assert len(F["y_train"]) == n_orig, "--repeat is only for a cache without paraphrases"
        for k in [k for k in F if k.startswith("train__")] + ["y_train"]:
            F[k] = np.concatenate([F[k]] * args.repeat)
        ytr = F["y_train"]
    key = f"{args.tag}_{meta.get('arm', 'orig')}_{args.dataset}_r{args.repeat}_s{args.seed}"
    ppath = args.out_dir / "preds_val" / f"{key}.npy"
    assert not ppath.exists(), f"{ppath} exists: one run per key"
    view = X.View(args.view, meta).fit({k: (v[:n_orig] if k.startswith("train__") else v) for k, v in F.items()})
    names = [re.sub(r"^existsT\[(\d+)\]", lambda m: f"some word ~ {{{meta['concepts'][int(m.group(1))]}}}", n) for n in view.names]
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    n_cl, ep = (200, 2) if args.smoke else (4000, 40)
    cfg = TMConfig(n_clauses=n_cl, T=n_cl, s=args.s, epochs=ep, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=2)
    tm = make_tm(C, Btr.shape[1], cfg, "half", device=dev)
    tm.fit(Btr, ytr, Bva, yva)
    pv = tm.predict(torch.as_tensor(Bva, dtype=torch.uint8, device=dev)).cpu().numpy()
    n_val, val_correct = len(yva), int((pv == yva).sum())
    assert val_correct == round(tm.best_val * n_val), f"restored state gives {val_correct} of {n_val}, best val {tm.best_val}"
    ppath.parent.mkdir(parents=True, exist_ok=True)
    with open(ppath, "xb") as f:  # exclusive: one file per run
        np.save(f, pv.astype(np.int16))
    hs = json.loads((EMB / "hardset.json").read_text())[args.dataset]
    rec = {"tag": args.tag, "study": "3d", "key": key, "smoke": bool(args.smoke), "repeat": args.repeat, "arm": meta.get("arm", "orig"),
           "n_orig": n_orig, "n_train": len(ytr), "s": args.s, "emb_dir": str(args.emb_dir), "table": meta.get("table"),
           "table_sha256": meta.get("table_sha256"), "kmeans_run_id": meta.get("kmeans_run_id"), "dataset": args.dataset,
           "view": args.view, "seed": args.seed, "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3),
           "cfg": asdict(cfg), "val_acc": round(tm.best_val, 4), "val_acc_check": round(float((pv == yva).mean()), 4),
           "n_val": n_val, "val_correct": val_correct, "preds_val": str(ppath),
           "best_epoch": getattr(tm, "best_epoch", None), "train_s": round(tm.train_seconds, 1),
           "lost_fixed": round(float((pv[hs["lost"]] == yva[hs["lost"]]).mean()), 4),
           "won_kept": round(float((pv[hs["won"]] == yva[hs["won"]]).mean()), 4),
           "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats(),
           "sha256": {"d3_exists_tm.py": sha16(Path(__file__).resolve()), "exists_tm.py": sha16(EMB / "exists_tm.py"),
                      "cache": sha16(cache)}}
    if "oos_val__mean" in F:
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device=dev)
        Xoo = torch.as_tensor(view.transform(F, "oos_val"), dtype=torch.uint8, device=dev)
        S = torch.cat([tm.class_sums(Xin), tm.class_sums(Xoo)]).cpu().numpy()
        is_oos = np.r_[np.zeros(len(Bva)), np.ones(len(F["oos_val__mean"]))]
        rec["oos_auroc_val_max_sum"] = round(X.auroc(is_oos, -np.sort(S, 1)[:, -1]), 4)
    vt = ds.texts["val"]
    rec["examples"] = {t: {"true": ds.intents[yva[vt.index(t)]], "tm": ds.intents[pv[vt.index(t)]]} for t in X.EXAMPLES if t in vt}
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["total_s"] = round(time.time() - t0, 1)
    with open(args.out_dir / "results_d3_val.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    st = tm.state()
    classes = [ds.intents.index(c) for c in X.SHOW.get(args.dataset, []) if c in ds.intents]
    lines = top_rules(st["ta"], st["w"], names, ds.intents, n_states=cfg.n_states, per_class=2, classes=classes)
    for t in X.EXAMPLES[:1]:
        if t in vt:
            lines.append(f"receipt for {t!r}:\n" + decision_receipt(st["ta"], st["w"], Bva[vt.index(t)], names, ds.intents,
                                                                     n_states=cfg.n_states, top=2))
    with open(args.out_dir / "rules_d3.txt", "a") as f:
        f.write(f"=== {key} {rec['table']} {args.view} s {args.s} val {rec['val_acc']}\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats", "examples")}))


if __name__ == "__main__":
    main()
