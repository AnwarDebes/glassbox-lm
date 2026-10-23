#!/usr/bin/env python
"""e82's R4 report items (no rule; dev only; CPU; read only): where the teacher-refined table changes the agent's
decisions, against the C4 table, at Stage A's settings, both seeds. Restored by logs/plan_e82_note2_2026-10-21.md.

    e82_breakdown.py --dataset star|abcd|multiwoz

For one dataset, it uses the 4 saved models: Stage A's 2 refined-table seeds (e82_<ds>_Bt_...) and the 2 C4 reference
seeds (the e74c / e78 / e77 / e79 rows of the plan).
- It rebuilds each view on the CPU: the C4 table through dl.count_text.patch, and the a3B table after
  scripts/newtable_run.register.
- It checks that each row's dev bits and CPU accuracy equal the row's (dl.cpu_tm_fast, integer votes).
- It averages each arm's per-turn correctness over its 2 seeds and reports refined minus C4: overall, first action
  (k = 0) against later ones, per gold label (the 20 most frequent), and for MultiWOZ "domain right" and "act right".
- It also gives per seed: both right, only the refined one right, and only the C4 one right.
Writes results/explore/e82_breakdown_<dataset>.json (refuses to overwrite). Added 2026-10-21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import count_text, cpu_tm_fast  # noqa: E402
from newtable_run import register  # noqa: E402

A3B = "c4words-a3B-v1"
A3B_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"
C4 = "c4words-S1000-v1"
ARMS = {"star": {"c4": [("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)], "a3b": [("e82_star_Bt_c16000_s16", 1), ("e82_star_Bt_c16000_s16", 2)]},
        "abcd": {"c4": [("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)], "a3b": [("e82_abcd_Bt_c4000_s16", 1), ("e82_abcd_Bt_c4000_s16", 2)]},
        "multiwoz": {"c4": [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)], "a3b": [("e82_multiwoz_Bt_c16000_s8", 1), ("e82_multiwoz_Bt_c16000_s8", 2)]}}


def doms(label: str) -> tuple:
    return tuple(sorted({p.split("-")[0] for p in label.split("+")})) if "-" in label else (label,)


def acts(label: str) -> tuple:
    return tuple(sorted({p.split("-", 1)[1] for p in label.split("+")})) if "-" in label else (label,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ARMS))
    args = ap.parse_args()
    ds = args.dataset
    out_path = ROOT / "results" / "explore" / f"e82_breakdown_{ds}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; not overwriting")
    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows[(d["tag"], d["seed"])] = d
    if register(Path("data/tables/c4words_a3B_v1"), A3B) != A3B_SHA:
        raise SystemExit("not the registered a3B table")
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", ds)
    dv, _ = load_examples("dev", ds)
    E = np.load(ROOT / "data" / "emb" / f"{ds}__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    y = np.array([r["y"] for r in dv])
    first = np.array([r["k"] == 0 for r in dv])
    gold = [blist[i] for i in y]
    corr, preds = {}, {}
    for arm, table in (("c4", C4), ("a3b", A3B)):
        count_text.patch(table)
        view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], blist)
        X = view.transform(dv, emb["dev"])
        h = hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()
        cs, ps = [], []
        for tag, seed in ARMS[ds][arm]:
            r = rows[(tag, seed)]
            if h != r["bits_sha256"]["dev"]:
                raise SystemExit(f"{tag} seed {seed}: dev bits {h[:8]} != the row's {r['bits_sha256']['dev'][:8]}")
            z = np.load(ROOT / "results" / "explore" / "abcd_models" / f"{tag}_seed{seed}.npz")
            p = cpu_tm_fast.class_sums(cpu_tm_fast.pack(z["ta"], z["w"]), X).argmax(1)
            acc = float((p == y).mean())
            if acc != r["acc"]:
                raise SystemExit(f"{tag} seed {seed}: CPU accuracy {acc} != the row's {r['acc']}")
            print(f"{arm} {tag} seed {seed}: bits {h[:8]} and accuracy {acc:.4f} equal the row's", flush=True)
            cs.append(p == y)
            ps.append(p)
        corr[arm] = np.mean(cs, 0)
        preds[arm] = ps
    d = corr["a3b"] - corr["c4"]
    out = {"what": "e82 R4 report items: refined (a3B) minus C4, dev, mean of 2 seeds per arm (no rule)", "dataset": ds, "n_dev": len(dv),
           "overall": round(100 * float(d.mean()), 3), "first_action": round(100 * float(d[first].mean()), 3),
           "later_actions": round(100 * float(d[~first].mean()), 3), "n_first": int(first.sum()), "per_seed_agreement": [], "per_label": {}}
    for i in range(2):
        a, c = preds["a3b"][i] == y, preds["c4"][i] == y
        out["per_seed_agreement"].append({"seed": i + 1, "both_right": round(float((a & c).mean()), 4), "only_refined": round(float((a & ~c).mean()), 4),
                                          "only_c4": round(float((~a & c).mean()), 4)})
    for lb, n in Counter(gold).most_common(20):
        m = np.array([g == lb for g in gold])
        out["per_label"][lb] = {"n": int(n), "c4": round(float(corr["c4"][m].mean()), 4), "refined": round(float(corr["a3b"][m].mean()), 4),
                                "diff": round(100 * float(d[m].mean()), 2)}
    if any("-" in g for g in gold):
        for name, fn in (("domain_right", doms), ("act_right", acts)):
            v = {}
            for arm in ("c4", "a3b"):
                v[arm] = float(np.mean([[fn(gold[j]) == fn(blist[p[j]]) for j in range(len(y))] for p in preds[arm]]))
            out[name] = {"c4": round(v["c4"], 4), "refined": round(v["a3b"], 4), "diff": round(100 * (v["a3b"] - v["c4"]), 2)}
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_label"}, indent=1))
    print("wrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
