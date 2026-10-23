#!/usr/bin/env python
"""R12 report: the English-trained readable layer on Norwegian (and English) test utterances.

Per utterance the mean correctness over TM seeds; composites = two disjoint 5-machine groups
(sum of class sums); paired bootstrap over utterances (10,000 resamples, seed 0, 95%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boot(d: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="R12")
    ap.add_argument("--encoder", default="bge-m3")
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration / args.encoder
    seeds = sorted(int(p.stem[4:]) for p in base.glob("seed*.json"))
    P = {k: np.load(base / f"preds_seed{k}.npz") for k in seeds}
    out = {"seeds": seeds}
    for lang in ("nb", "en"):
        y = P[seeds[0]][f"y_{lang}"]
        corr = np.stack([P[k][f"S_{lang}"].argmax(1) == y for k in seeds]).astype(float)
        lr_b = np.load(base / "lr_bits_preds.npz")
        lr_r = np.load(base / "lr_raw_preds.npz")
        rule = np.load(base / "rule_preds.npz")
        cb = (lr_b[f"pred_{lang}"] == y).astype(float)
        cr = (lr_r[f"pred_{lang}"] == y).astype(float)
        cu = (rule[f"pred_{lang}"] == y).astype(float)
        comps = []
        for grp in (seeds[:5], seeds[5:10]):
            if len(grp) == 5:
                S = sum(P[k][f"S_{lang}"].astype(np.float64) for k in grp)
                comps.append(S.argmax(1) == y)
        r = {"tm_acc_mean": float(corr.mean()), "tm_acc_per_seed": [round(float(a), 5) for a in corr.mean(1)],
             "lr_bits_acc": float(cb.mean()), "lr_raw_acc": float(cr.mean()), "rule_acc": float(cu.mean()),
             "tm_minus_lr_bits": boot(corr.mean(0) - cb), "tm_minus_lr_raw": boot(corr.mean(0) - cr),
             "tm_agrees_with_rule": float(np.mean([np.mean(P[k][f"S_{lang}"].argmax(1) == rule[f"pred_{lang}"]) for k in seeds]))}
        if comps:
            cc = np.stack(comps).astype(float)
            r["composite_acc"] = [float(a) for a in cc.mean(1)]
            r["composite_minus_lr_raw"] = boot(cc.mean(0) - cr)
        out[lang] = r
    out["retention_nb_over_en"] = out["nb"]["tm_acc_mean"] / out["en"]["tm_acc_mean"]
    out["R12.1"] = out["nb"]["tm_minus_lr_bits"]["diff"] >= -0.005
    out["R12.2"] = out["nb"].get("composite_minus_lr_raw", {"diff": -1})["diff"] >= -0.010
    out["R12.3"] = out["retention_nb_over_en"] >= 0.90
    out["R12_met"] = bool(out["R12.1"] and out["R12.2"] and out["R12.3"])
    (base.parent / "report_R12.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
