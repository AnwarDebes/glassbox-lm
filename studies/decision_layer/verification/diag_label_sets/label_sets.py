"""Where the TM's MultiWOZ errors fall against LR's on the same bits, as label SETS (dev only; CPU; read only; no rule).

    cd <study> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python -I <this file> <out.json>

Private diagnostic (2026-10-21, while e81 waits for the GPU; outside the repository). MultiWOZ's 128 labels are sets of
domain-act parts joined by "+" (for example general-bye+general-welcome), plus OTHER. For the C4 table and the refined
(a3B) table, it rebuilds view B on the CPU, checks the dev bits against the rows, rebuilds the TM's decisions for the
two 16k seeds (checked against each row's accuracy) and reads LR's saved dev predictions (C chosen on dev). For each
predictor it reports:
- exact accuracy, part-level micro precision / recall / F1, and the mean Jaccard of predicted and gold part sets;
- every wrong turn as subset (prediction inside gold), superset, overlap (shares a part, neither), or disjoint;
- on the turns LR gets right and the TM gets wrong, the TM's error kinds.
If the TM's extra errors are mostly near misses (subset/superset/overlap), a reader that predicts the parts may help.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

S = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(S / "src"))
sys.path.insert(1, str(S / "scripts"))

from dl import count_text, cpu_tm_fast  # noqa: E402
from newtable_run import register  # noqa: E402

A3B_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"
ARMS = {"C4": {"table": "c4words-S1000-v1", "tm": [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)], "lr": "dry_e77mwoz_lr_mem_c4"},
        "a3B": {"table": "c4words-a3B-v1", "tm": [("e82_multiwoz_Bt_c16000_s8", 1), ("e82_multiwoz_Bt_c16000_s8", 2)], "lr": "dry_e82mwoz_lr_mem_a3B"}}


def parts(label: str) -> frozenset:
    return frozenset(label.split("+"))


def kinds(pred, y, blist) -> dict:
    out = {"exact": 0.0, "subset": 0.0, "superset": 0.0, "overlap": 0.0, "disjoint": 0.0}
    tp = fp = fn = 0
    jac = []
    for p, g in zip(pred, y):
        P, G = parts(blist[p]), parts(blist[g])
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
        jac.append(len(P & G) / len(P | G))
        k = "exact" if P == G else "subset" if P < G else "superset" if P > G else "overlap" if P & G else "disjoint"
        out[k] += 1
    n = len(y)
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return {**{k: round(v / n, 4) for k, v in out.items()}, "part_precision": round(prec, 4), "part_recall": round(rec, 4),
            "part_f1": round(2 * prec * rec / (prec + rec), 4), "jaccard": round(float(np.mean(jac)), 4)}


def main() -> None:
    out_path = Path(sys.argv[1])
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; not overwriting")
    rows = {}
    for line in (S / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)
    if register(S / "data" / "tables" / "c4words_a3B_v1", "c4words-a3B-v1") != A3B_SHA:
        raise SystemExit("not the registered a3B table")
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", "multiwoz")
    dv, _ = load_examples("dev", "multiwoz")
    E = np.load(S / "data" / "emb" / "multiwoz__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    y = np.array([r["y"] for r in dv])
    res = {"what": "MultiWOZ dev: TM and LR errors as label sets (private diagnostic; no rule)", "n_dev": len(y),
           "gold_parts_per_label": dict(Counter(len(parts(blist[g])) for g in y)), "arms": {}}
    for arm, a in ARMS.items():
        count_text.patch(a["table"])
        view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], blist)
        X = view.transform(dv, emb["dev"])
        h = hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()
        L = np.load(S / "results" / "explore" / "dryrun" / a["lr"] / "seed0.npz")
        lj = json.loads((S / "results" / "explore" / "dryrun" / a["lr"] / "seed0.json").read_text())
        if not np.array_equal(L["y"], y) or lj["bits_sha256"]["dev"] != h or float((L["pred"] == y).mean()) != lj["dev_acc"]:
            raise SystemExit(f"{arm}: LR file does not match the dev rows or bits")
        r_arm = {"LR": kinds(L["pred"], y, blist), "LR_acc": lj["dev_acc"], "TM": []}
        for tag, seed in a["tm"]:
            (row,) = rows[(tag, seed)]
            if h != row["bits_sha256"]["dev"]:
                raise SystemExit(f"{tag} seed {seed}: dev bits differ")
            z = np.load(S / "results" / "explore" / "abcd_models" / f"{tag}_seed{seed}.npz")
            p = cpu_tm_fast.class_sums(cpu_tm_fast.pack(z["ta"], z["w"]), X).argmax(1)
            if float((p == y).mean()) != row["acc"]:
                raise SystemExit(f"{tag} seed {seed}: CPU accuracy differs from the row")
            only_lr = (L["pred"] == y) & (p != y)
            k_only = kinds(p[only_lr], y[only_lr], blist)
            r_arm["TM"].append({"tag": tag, "seed": seed, "acc": row["acc"], **kinds(p, y, blist),
                                "on_lr_right_tm_wrong": {"share_of_dev": round(float(only_lr.mean()), 4),
                                                         **{k: v for k, v in k_only.items() if k in ("subset", "superset", "overlap", "disjoint")}}})
            print(arm, tag, seed, "checked", flush=True)
        res["arms"][arm] = r_arm
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
