#!/usr/bin/env python
"""Per-class correction of the TM's votes on STAR, post hoc on R18's saved models (DEV only).

Diagnosis d1 (results/explore/star_gap.jsonl): on dev the TM's gap to LR sits in the less common
actions (21 to 400 train examples: 7 to 9 points behind), it is level on the common ones, and a
class's positive weight mass grows with its train count. Does a per-class correction close it?

- ``center``: S_c minus the mean of S_c over the train inputs;
- ``zscore``: (S_c - mean_c) / std_c over the train inputs;
- ``mass``: S_c / (positive weight mass of c) ** tau;
- ``prior``: S_c - tau * sd(S) * log(train count of c) (logit adjustment, Menon et al. 2021).

tau is picked on one half of the dev conversations and scored on the other (2 folds, both ways);
the full-dev grid is printed too. One JSON line to results/explore/star_gap.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.bitcache import view_bits  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def load_tm(path: Path, n_classes: int, n_bits: int) -> CoTM:
    m = np.load(path)
    tm = CoTM(n_classes, n_bits, TMConfig(**json.loads(str(m["cfg"]))))
    tm.ta = torch.as_tensor(m["ta"].astype(np.int32), device="cuda")
    tm.w = torch.as_tensor(m["w"].astype(np.int32), device="cuda")
    return tm


def main() -> None:
    Xtr, ytr, Xdv, ydv, names, blist, tr, dv = view_bits("star")
    src = ROOT / "results" / "confirm" / "R18" / "tm_mem"
    assert names == json.loads((src / "bit_names.json").read_text())
    C_ = len(blist)
    cnt = np.bincount(ytr, minlength=C_).astype(float)
    fold = np.array([int(hashlib.sha256(str(r["convo_id"]).encode()).hexdigest(), 16) % 2 for r in dv])
    Xt = torch.as_tensor(Xtr, dtype=torch.uint8, device="cuda")
    Xd = torch.as_tensor(Xdv, dtype=torch.uint8, device="cuda")
    rec = {"tag": "d2_star_vote_norm", "split": "dev", "taus": TAUS, "seeds": {}}
    for k in range(1, 6):
        tm = load_tm(src / f"model_seed{k}.npz", C_, Xdv.shape[1])
        St = tm.class_sums(Xt).cpu().numpy()
        Sd = tm.class_sums(Xd).cpu().numpy()
        mu, sd = St.mean(0), St.std(0) + 1e-6
        mass = np.clip(tm.w.cpu().numpy(), 0, None).sum(0).astype(float) + 1.0
        scale = Sd.std()
        variants = {
            "raw": {0.0: Sd},
            "center": {t: Sd - t * mu for t in TAUS},
            "zscore": {t: (Sd - t * mu) / sd ** min(t, 1.0) for t in TAUS},
            "mass": {t: Sd / mass ** t for t in TAUS},
            "prior": {t: Sd - t * scale * 0.1 * np.log(cnt + 1) for t in TAUS},
        }
        out = {}
        for name, by_t in variants.items():
            acc = {t: (S.argmax(1) == ydv) for t, S in by_t.items()}
            full = {str(t): float(a.mean()) for t, a in acc.items()}
            held = []
            for f in (0, 1):
                best_t = max(acc, key=lambda t: acc[t][fold == f].mean())
                held.append((float(acc[best_t][fold != f].mean()), best_t, int((fold != f).sum())))
            n0, n1 = held[0][2], held[1][2]
            out[name] = {"full_dev_by_tau": full, "two_fold": float((held[0][0] * n0 + held[1][0] * n1) / (n0 + n1)),
                         "picked": [h[1] for h in held]}
        rec["seeds"][k] = out
        print(k, {n: (round(v["two_fold"], 4), v["picked"]) for n, v in out.items()}, flush=True)
    rec["mean_two_fold"] = {n: float(np.mean([rec["seeds"][k][n]["two_fold"] for k in rec["seeds"]])) for n in rec["seeds"][1]}
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (ROOT / "results" / "explore" / "star_gap.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec["mean_two_fold"]))


if __name__ == "__main__":
    main()
