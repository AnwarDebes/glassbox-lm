#!/usr/bin/env python
"""Cross-lingual readable decision layer: trained on English only, used on Norwegian (MASSIVE).

Everything is fitted on MASSIVE en-US: the readable view (whitened intent centres of the English
training queries, `proto:k=1:whiten=0.001`, no words: English words do not transfer), the TM
(best snapshot on English dev), logistic regression (C from English dev). The same models then
route the nb-NO translation of the queries, embedded by the same multilingual encoder. No
Norwegian data is used for any fitting or choice.

    xling_run.py --encoder bge-m3 --eval dev --seeds 1            (exploration: dev splits)
    xling_run.py --encoder bge-m3 --eval test --registration R12 --seeds 1 2 ...  (registered)

Per seed: the TM's predictions and class sums on the English and the Norwegian eval split.
Once per run: LR on the same bits, LR on the raw embedding, the nearest-centre rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

GRID_C = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--view", default="proto:k=1:whiten=0.001")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--registration", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.eval == "test" and not args.registration:
        raise SystemExit("test split only in registered runs")
    ev = {"dev": "val", "test": "test"}[args.eval]

    en, nb = load("massive_en"), load("massive_nb")
    assert en.intents == nb.intents
    de = np.load(ROOT / "data" / "emb" / f"massive_en__{args.encoder}.npz")
    dn = np.load(ROOT / "data" / "emb" / f"massive_nb__{args.encoder}.npz")
    C_ = len(en.intents)
    ytr, yva = en.labels["train"], en.labels["val"]
    y_en, y_nb = en.labels[ev], nb.labels[ev]
    view = make_view(args.view, seed=0)
    view.fit(de["train"], ytr, texts=en.texts["train"], class_names=en.intents)
    Xtr, Xva = view.transform(de["train"], en.texts["train"]), view.transform(de["val"], en.texts["val"])
    X_en, X_nb = view.transform(de[ev], en.texts[ev]), view.transform(dn[ev], nb.texts[ev])

    if args.registration:
        out = ROOT / "results" / "confirm" / args.registration / args.encoder
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = None
    rec_run = {"tag": args.tag, "encoder": args.encoder, "view": args.view, "eval": args.eval, "n_bits": int(Xtr.shape[1]),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # once per run: linear models and the one rule (deterministic)
    base_path = (out / "baselines.json") if out else None
    if out is None or not base_path.exists():
        res = {}
        for name, (A, B_en, B_nb, Va) in {
            "lr_bits": (Xtr, X_en, X_nb, Xva),
            "lr_raw": (de["train"], de[ev], dn[ev], de["val"]),
        }.items():
            best = None
            for c in GRID_C:
                W, b = fit_logreg(A, ytr, C_, c)
                acc = float((logits(Va, W, b).argmax(1).cpu().numpy() == yva).mean())
                if best is None or acc > best[0]:
                    best = (acc, c, W, b)
            _, c, W, b = best
            p_en = logits(B_en, W, b).argmax(1).cpu().numpy()
            p_nb = logits(B_nb, W, b).argmax(1).cpu().numpy()
            res[name] = {"C": c, "acc_en": float((p_en == y_en).mean()), "acc_nb": float((p_nb == y_nb).mean())}
            if out:
                np.savez(out / f"{name}_preds.npz", pred_en=p_en, pred_nb=p_nb, y_en=y_en, y_nb=y_nb)
        proto = view.views[0] if hasattr(view, "views") else view
        r_en = proto.sims(de[ev]).argmax(1).cpu().numpy()
        r_nb = proto.sims(dn[ev]).argmax(1).cpu().numpy()
        res["rule_nearest_centre"] = {"acc_en": float((r_en == y_en).mean()), "acc_nb": float((r_nb == y_nb).mean())}
        if out:
            np.savez(out / "rule_preds.npz", pred_en=r_en, pred_nb=r_nb, y_en=y_en, y_nb=y_nb)
            base_path.write_text(json.dumps({**rec_run, **res}, indent=1))
        else:
            with (ROOT / "results" / "explore" / "xling.jsonl").open("a") as f:
                f.write(json.dumps({**rec_run, "arm": "baselines", **res}) + "\n")
        print(json.dumps(res), flush=True)

    for seed in args.seeds:
        if out and (out / f"seed{seed}.json").exists():
            print("exists seed", seed)
            continue
        cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.epochs, seed=seed)
        tm = make_tm(C_, Xtr.shape[1], cfg, init="half")
        tm.fit(Xtr, ytr, Xva, yva)
        S_en = tm.class_sums(torch.as_tensor(X_en, dtype=torch.uint8, device="cuda")).cpu().numpy()
        S_nb = tm.class_sums(torch.as_tensor(X_nb, dtype=torch.uint8, device="cuda")).cpu().numpy()
        rec = {**rec_run, "seed": seed, "cfg": asdict(cfg), "best_epoch": getattr(tm, "best_epoch", None),
               "acc_en": float((S_en.argmax(1) == y_en).mean()), "acc_nb": float((S_nb.argmax(1) == y_nb).mean()),
               "stats": tm.stats(), "train_s": round(tm.train_seconds, 1)}
        if out:
            np.savez_compressed(out / f"preds_seed{seed}.npz", S_en=S_en.astype(np.int32), S_nb=S_nb.astype(np.int32),
                                y_en=y_en, y_nb=y_nb)
            (out / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
        else:
            with (ROOT / "results" / "explore" / "xling.jsonl").open("a") as f:
                f.write(json.dumps({**rec, "arm": "tm"}) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg",)}), flush=True)


if __name__ == "__main__":
    main()
