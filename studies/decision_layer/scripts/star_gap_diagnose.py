#!/usr/bin/env python
"""Why is the TM 3 points behind LR on STAR (R18)? Diagnosis on DEV only (the test split is not read).

Uses R18's saved TM-mem models (results/confirm/R18/tm_mem/model_seed<k>.npz, best-dev snapshots)
and an LR fitted here on the same bits. Reports, on dev:
- accuracy of each seed, and of the 5 seeds' summed class sums (is the gap variance or bias?);
- ties at the top of the TM's integer votes; the rank of the true class;
- accuracy by how common the true action is in train (rare-class deficit?);
- where the two disagree; how many clauses fire per input.
Writes one JSON line to results/explore/star_gap.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

GRID_C = [0.01, 0.03, 0.1, 0.3, 1.0]


def load_tm(path: Path, n_classes: int, n_bits: int) -> CoTM:
    m = np.load(path)
    tm = CoTM(n_classes, n_bits, TMConfig(**json.loads(str(m["cfg"]))))
    tm.ta = torch.as_tensor(m["ta"].astype(np.int32), device="cuda")
    tm.w = torch.as_tensor(m["w"].astype(np.int32), device="cuda")
    return tm


def main() -> None:
    tr, blist = load_examples("train", "star")
    dv, _ = load_examples("dev", "star")
    E = np.load(ROOT / "data" / "emb" / "star__bge-large.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "intent", "step", "words"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    src = ROOT / "results" / "confirm" / "R18" / "tm_mem"
    assert view.names == json.loads((src / "bit_names.json").read_text())
    Xtr, Xdv = view.transform(tr, em["train"]), view.transform(dv, em["dev"])
    ytr, ydv = np.array([r["y"] for r in tr]), np.array([r["y"] for r in dv])
    C_ = len(blist)
    rec: dict = {"tag": "d1_star_gap", "split": "dev", "n_dev": len(dv), "n_classes": C_}

    # LR, with a grid reaching below R18's (R18 used 0.1 to 100 and picked 0.1, the edge)
    lr = {}
    for c in GRID_C:
        W, b = fit_logreg(Xtr, ytr, C_, c)
        lr[c] = logits(Xdv, W, b).cpu().numpy()
    rec["lr_dev_acc_by_C"] = {str(c): float((v.argmax(1) == ydv).mean()) for c, v in lr.items()}
    Slr = lr[0.1]  # R18's choice
    p_lr = Slr.argmax(1)

    Xd = torch.as_tensor(Xdv, dtype=torch.uint8, device="cuda")
    sums, fires = [], []
    for k in range(1, 6):
        tm = load_tm(src / f"model_seed{k}.npz", C_, Xdv.shape[1])
        sums.append(tm.class_sums(Xd).cpu().numpy())
        fires.append(tm.clause_outputs(Xd).sum(1).float().cpu().numpy())
    rec["tm_dev_acc_by_seed"] = [float((S.argmax(1) == ydv).mean()) for S in sums]
    Sens = np.sum(sums, axis=0)
    rec["tm_dev_acc_5seed_vote_sum"] = float((Sens.argmax(1) == ydv).mean())
    S = sums[0]
    p_tm = S.argmax(1)
    ok_tm, ok_lr = p_tm == ydv, p_lr == ydv

    srt = np.sort(S, axis=1)
    rec["tm_top_tie_rate"] = float((srt[:, -1] == srt[:, -2]).mean())
    true_s = S[np.arange(len(ydv)), ydv]
    rec["tm_true_ties_top_but_lost"] = float(((true_s == srt[:, -1]) & ~ok_tm).mean())
    rank_tm = (S > true_s[:, None]).sum(1)
    rank_lr = (Slr > Slr[np.arange(len(ydv)), ydv][:, None]).sum(1)
    rec["tm_true_in_top3"] = float((rank_tm < 3).mean())
    rec["lr_true_in_top3"] = float((rank_lr < 3).mean())
    rec["tm_max_sum_le_0"] = float((srt[:, -1] <= 0).mean())
    rec["clauses_firing_per_input"] = {"mean": float(fires[0].mean()), "p10": float(np.percentile(fires[0], 10)),
                                       "p50": float(np.median(fires[0])), "n_clauses": int(sums[0].shape[1] and 16000)}
    rec["both_right"] = float((ok_tm & ok_lr).mean())
    rec["tm_only"] = float((ok_tm & ~ok_lr).mean())
    rec["lr_only"] = float((~ok_tm & ok_lr).mean())

    cnt = np.bincount(ytr, minlength=C_)
    bins = [(1, 20), (21, 100), (101, 400), (401, 1500), (1501, 10**9)]
    rec["by_train_freq"] = []
    for lo, hi in bins:
        m = (cnt[ydv] >= lo) & (cnt[ydv] <= hi)
        if m.sum() == 0:
            continue
        rec["by_train_freq"].append({"train_count": f"{lo}-{hi}", "n_dev": int(m.sum()), "n_classes": int(((cnt >= lo) & (cnt <= hi)).sum()),
                                     "tm": float(ok_tm[m].mean()), "lr": float(ok_lr[m].mean()),
                                     "tm_5seed": float((Sens.argmax(1) == ydv)[m].mean())})
    # how the TM spreads its errors: predicted class frequency vs true
    rec["tm_pred_distinct"] = int(len(np.unique(p_tm)))
    rec["lr_pred_distinct"] = int(len(np.unique(p_lr)))
    rec["true_distinct"] = int(len(np.unique(ydv)))
    # per-class weight mass of the TM
    w = tm.w.cpu().numpy()  # seed 5's model; mass per class
    mass = np.clip(w, 0, None).sum(0)
    rec["tm_pos_weight_mass_by_train_freq"] = {f"{lo}-{hi}": float(mass[(cnt >= lo) & (cnt <= hi)].mean()) for lo, hi in bins if ((cnt >= lo) & (cnt <= hi)).any()}
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (ROOT / "results" / "explore" / "star_gap.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
