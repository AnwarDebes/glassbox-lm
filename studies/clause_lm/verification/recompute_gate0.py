#!/usr/bin/env python
"""Independent recompute of the clause_lm Gate 0 cause check (plan docs/plan_gate0_cause_2026-10-25.md, frozen
2026-10-25T12:59:23Z), written from the plan text without the driver (src/gate0_cause.py), its checker or the
upstream package.

For each saved model (tm.npz) it recomputes, from the model arrays and the feature file only:
  - the validation accuracy: the bits are a thermometer code with one threshold per feature fitted on Xtrain
    (median of the column; the midpoint when the column holds only its minimum and maximum; minimum + 1 when the
    column is constant), bit = value >= threshold; a clause fires when every included literal is 1 and it includes
    at least one (include iff ta > n_states; literal k < f is bit k, literal k >= f is NOT bit k - f); a suppressed
    clause never fires; the class is the first argmax of the summed weights of the firing clauses;
  - the literal statistics the plan's signatures use;
and it applies the plan's signatures, the three verdicts and the ordered outcome table to the recomputed numbers.
Only Xtrain, Xval and yval are read from the feature file (allow_pickle=False; no test array is loaded).

Usage (from any folder; numpy only):
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 19 \
    studies/decision_layer/.venv-dl/bin/python -I recompute_gate0.py OUT.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import glob
import hashlib
import json
import os
import sys

import numpy as np

STUDY = f"{_REPO_ROOT}/studies/clause_lm"
EMB = os.path.join(STUDY, "work/m5/emb_sorted/clinc150__lexical.npz")
FREEZE_RECORD = os.path.join(STUDY, "docs/plan_gate0_cause_2026-10-25.frozen.json")
RECORDED_A0 = 0.8186666666666667  # the plan's recorded torch value (2026-10-14, seed 42)

# Counting runs named by the plan's job lines (locked settings), and the two reference models the plan quotes.
RUNS = {
    "torch_seed42": "results/20261025T130431Z_gate0_cause_torch_s8_seed42",
    "torch_noneg_seed42": "results/20261025T130626Z_gate0_cause_torch_noneg_s8_seed42",
    "torch_noneg_seed123": "results/20261025T130751Z_gate0_cause_torch_noneg_s8_seed123",
    "torch_noneg_seed456": "results/20261025T130918Z_gate0_cause_torch_noneg_s8_seed456",
}
SMOKE = "results/20261025T130238Z_gate0_cause_torch_noneg_s8_seed7"
REFERENCES = {
    "numba_gate0_2026-10-02": ("work/m5/runs/sorted/s8_seed42/clinc150__lexical__seed42/tm.npz", 0.5493333333333333),
    "torch_gate0_2026-10-14": ("work/m5/runs/sorted_torchgpu/s8_seed42/clinc150__lexical__seed42/tm.npz",
                               0.8186666666666667),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thresholds(Xtrain):
    """One threshold per column, from the plan's description of the 2-bin thermometer code."""
    X = np.asarray(Xtrain, dtype=np.float32)
    lo, hi = X.min(axis=0), X.max(axis=0)
    th = np.empty(X.shape[1], dtype=np.float32)
    kinds = {"constant": 0, "two_valued": 0, "median": 0}
    for d in range(X.shape[1]):
        col = X[:, d]
        if hi[d] <= lo[d]:
            th[d] = np.float32(lo[d] + np.float32(1.0))
            kinds["constant"] += 1
        elif np.all((col == lo[d]) | (col == hi[d])):
            th[d] = np.float32((lo[d] + hi[d]) / np.float32(2.0))
            kinds["two_valued"] += 1
        else:
            th[d] = np.float32(np.percentile(col, 50.0))
            kinds["median"] += 1
    return th, kinds


def booleanize(X, th):
    return (np.asarray(X, dtype=np.float32) >= th[None, :]).astype(np.uint8)


def clause_fire(ta, n_states, bits, suppressed=None):
    """[n_samples, n_clauses] bool: every included literal is 1 and at least one is included."""
    inc = np.asarray(ta) > n_states
    f = bits.shape[1]
    if inc.shape[1] != 2 * f:
        raise ValueError(f"model has {inc.shape[1]} literals, the data {2 * f}")
    lits = np.concatenate([bits, 1 - bits], axis=1).astype(np.float32)  # [B, 2f]
    zeros_hit = (1.0 - lits) @ inc.T.astype(np.float32)  # included literals that are 0, per sample and clause
    fire = (zeros_hit == 0.0) & (inc.sum(axis=1)[None, :] > 0)
    if suppressed is not None:
        fire &= ~np.asarray(suppressed, dtype=bool)[None, :]
    return fire


def predict(ta, weight, n_states, bits, suppressed=None):
    fire = clause_fire(ta, n_states, bits, suppressed)
    sums = fire.astype(np.int64) @ np.asarray(weight, dtype=np.int64)
    pred = np.argmax(sums, axis=1)
    top = sums.max(axis=1)
    ties = int(((sums == top[:, None]).sum(axis=1) > 1).sum())
    return pred, ties


def literal_stats(ta, weight, n_states):
    inc = np.asarray(ta) > n_states
    f = inc.shape[1] // 2
    pos, neg = inc[:, :f].sum(1), inc[:, f:].sum(1)
    tot = pos + neg
    w = np.asarray(weight)
    return {"n_clauses": int(inc.shape[0]), "clauses_with_positive_literal": int((pos > 0).sum()),
            "clauses_with_negated_literal": int((neg > 0).sum()), "non_empty_clauses": int((tot > 0).sum()),
            "literals_median": float(np.median(tot)), "literals_mean": float(tot.mean()),
            "literals_max": int(tot.max()), "positive_median": float(np.median(pos)), "positive_max": int(pos.max()),
            "negated_median": float(np.median(neg)), "negated_max": int(neg.max()),
            "weight_negative_share": float((w < 0).mean()), "weight_zero_share": float((w == 0).mean())}


def signature(acc, st):
    if acc <= 0.70 and st["clauses_with_positive_literal"] <= 400 and st["literals_median"] >= 100:
        return "degenerate"
    if acc >= 0.78 and st["clauses_with_positive_literal"] >= 2000 and st["literals_median"] <= 10:
        return "healthy"
    return "neither"


def verdicts(runs, arm_b_state):
    """The plan's three verdicts and the ordered outcome table, from recomputed rows."""
    a0 = runs.get("torch_seed42")
    if a0 is None:
        driver = "MISSING"
    else:
        ok = abs(a0["val_acc"] - RECORDED_A0) <= 0.01 and a0["stats"]["clauses_with_positive_literal"] >= 2000
        driver = "MET" if ok else "NOT MET"
    sig_a = [runs[k]["signature"] for k in ("torch_noneg_seed42", "torch_noneg_seed123", "torch_noneg_seed456")
             if k in runs]
    if len(sig_a) < 3:
        arm_a = "MISSING"
    elif all(s == "degenerate" for s in sig_a):
        arm_a = "CONFIRMED"
    elif all(s == "healthy" for s in sig_a):
        arm_a = "REFUTED"
    else:
        arm_a = "MIXED"
    arm_b = arm_b_state
    if driver != "MET":
        row = 1
    elif arm_a in ("MIXED", "MISSING") or arm_b in ("MIXED", "MISSING"):
        row = 2
    elif "REFUTED" in (arm_a, arm_b):
        row = 3
    elif arm_a == "CONFIRMED" and arm_b == "CONFIRMED":
        row = 4
    else:
        row = 5
    return {"driver_check": driver, "arm_A": arm_a, "arm_B": arm_b, "outcome_row": row}


def evaluate(model_path, Bval, yval):
    with np.load(model_path, allow_pickle=False) as z:
        ta, weight = z["ta"], z["weight"]
        n_states = int(z["cfg_n_states"])
        suppressed = z["suppressed"] if "suppressed" in z.files else None
    pred, ties = predict(ta, weight, n_states, Bval, suppressed)
    acc = float((pred == np.asarray(yval)).mean())
    st = literal_stats(ta, weight, n_states)
    return {"file": model_path, "sha256": sha256(model_path), "val_acc": acc, "top_ties": ties,
            "suppressed_true": int(np.asarray(suppressed).sum()) if suppressed is not None else None,
            "n_states": n_states, "stats": st, "signature": signature(acc, st)}


def main(out_path):
    problems = []
    with np.load(EMB, allow_pickle=False) as e:
        Xtrain, Xval, yval = e["Xtrain"], e["Xval"], e["yval"]
    th, kinds = thresholds(Xtrain)
    Bval = booleanize(Xval, th)
    freeze_utc = json.load(open(FREEZE_RECORD))["frozen_utc"]
    out = {"emb": EMB, "emb_sha256": sha256(EMB), "threshold_kinds": kinds, "freeze_utc": freeze_utc,
           "val_rows": int(len(yval)), "runs": {}, "references": {}, "smoke": None}
    for key, folder in RUNS.items():
        d = os.path.join(STUDY, folder)
        row = evaluate(os.path.join(d, "tm.npz"), Bval, yval)
        m = json.load(open(os.path.join(d, "metrics.json")))
        row["metrics_val_acc"] = m["val_acc"]
        row["val_acc_equal"] = row["val_acc"] == m["val_acc"]
        row["stats_equal"] = {k: row["stats"][k] == m["literal_stats"][k] for k in row["stats"]}
        start = os.path.basename(folder)[:16]
        row["start_utc"] = f"{start[:4]}-{start[4:6]}-{start[6:8]}T{start[9:11]}:{start[11:13]}:{start[13:15]}Z"
        row["after_freeze"] = row["start_utc"] > freeze_utc
        if not row["val_acc_equal"]:
            problems.append(f"{key}: recomputed val_acc {row['val_acc']} != metrics {m['val_acc']}")
        if not all(row["stats_equal"].values()):
            problems.append(f"{key}: literal stats differ: {[k for k, v in row['stats_equal'].items() if not v]}")
        if not row["after_freeze"]:
            problems.append(f"{key}: started {row['start_utc']}, not after the freeze {freeze_utc}")
        out["runs"][key] = row
    for key, (rel, recorded) in REFERENCES.items():
        row = evaluate(os.path.join(STUDY, rel), Bval, yval)
        row["recorded_val_acc"] = recorded
        row["val_acc_equal"] = row["val_acc"] == recorded
        if not row["val_acc_equal"]:
            problems.append(f"reference {key}: recomputed {row['val_acc']} != recorded {recorded}")
        out["references"][key] = row
    sm = os.path.join(STUDY, SMOKE)
    with open(os.path.join(sm, "config.json")) as f:
        cfg = json.load(f)
    out["smoke"] = {"folder": SMOKE, "epochs": cfg.get("epochs"), "seed": cfg.get("seed"),
                    "locked": cfg.get("locked_settings_of_m5_step1")}
    erode = sorted(glob.glob(os.path.join(STUDY, "results/*gate0_cause_numba_erode*")))
    locked_erode = []
    for d in erode:
        try:
            if json.load(open(os.path.join(d, "config.json"))).get("locked_settings_of_m5_step1"):
                locked_erode.append(d)
        except OSError:
            pass
    arm_b = "NOT RUN" if not locked_erode else "SEE FOLDERS"
    out["arm_B_folders"] = {"all": [os.path.basename(d) for d in erode], "locked": locked_erode}
    out["verdicts"] = verdicts(out["runs"], arm_b)
    out["problems"] = problems
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"verdicts": out["verdicts"], "problems": problems,
                      "val_acc": {k: v["val_acc"] for k, v in out["runs"].items()},
                      "references": {k: v["val_acc"] for k, v in out["references"].items()},
                      "signatures": {k: v["signature"] for k, v in out["runs"].items()},
                      "threshold_kinds": kinds}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
