"""EXPLORATORY tuning of the hierarchical package on the H-AO world (docs/13 A7, H17.5), seed 100 only.

Why this exists: the exploratory seed-100 run of experiments/ssl_r2_hao.py (results/20260924T215754Z_ssl_r2_hao_gpu_
exploratory/) left every package model (hierarchical and flat, negated and positive literals) at training accuracy
0.5000, while the package's own noisy-parity control passed and gradient-boosted trees reached 1.00. This script
tries package configurations on the TUNING split only and records, per configuration, a training curve with
diagnostics, so the reason for the failure can be read off and a configuration that trains can be chosen.

Data (identical to the fit/val bags of experiments/ssl_r2_hao.py, checked by sha256 against the exploratory run):
    world      hao_world.make_world(seed) with the config's world parameters
    fit bags   n_fit bags from the fit combinations (carve_combos of the TRAINING combinations, stream 5)
    val bags   n_val bags from the validation combinations carved from the TRAINING combinations
The held-out TEST combinations are never sampled and never scored here.

Models: exactly the constructors of experiments/ssl_r2_hao.py (pkg_make: TsetlinMachine of the installed package,
grid (208,1,1), block (128,1,1), seed = world seed), so every chosen configuration is expressible in that script's
config schema (kind, clauses, alternatives, n_concepts, T, s, state_bits, boost, append_negated, weighted_clauses).

Per configuration and evaluation epoch: training accuracy (fit bags), validation accuracy, share predicted
positive, share of class sums exactly 0 (the single-output machine predicts POSITIVE at class sum >= 0), mean
clipped class sum per class, leaf statistics from the TA include bits (share of empty leaves per clause polarity,
included positive literals of group words and distractors, included negated literals, per leaf), group-word
coverage of positive-polarity clauses, and for the hierarchical kind the OR-group recovery (Jaccard, as in the
main script).

Usage (detached, CPU slot first, docs/12 F25):
    setsid nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/mem_guard.py \\
        --scope group --limit-gib 8 -- scripts/run_logged.py --name ssl_r2_hao_tune_<round>_exploratory \\
        --config <tuning config> -- .venv-tm/bin/python -u experiments/ssl_r2_hao_tune.py --config <tuning config> \\
        [--only name1,name2] > /dev/null 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import hao_world as H  # noqa: E402

import ssl_r2_hao as E  # noqa: E402  (constructors, include decoding, stream ids)

T0 = time.time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}


def flush() -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_metrics(METRICS, on_conflict="replace")


def out_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path("/tmp") / f"ssl_r2_hao_tune_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def tuning_data(seed: int, W: dict) -> dict:
    world = H.make_world(seed, vocab=int(W["vocab"]), n_groups=int(W["n_groups"]), group_size=int(W["group_size"]),
                         train_fraction=float(W["train_fraction"]))
    kw = {"n_distractors": int(W["n_distractors"]), "extra_probs": tuple(W["extra_probs"])}
    fit_mask, val_mask = H.carve_combos(world, world.train_combo, float(W["val_fraction"]), E.STREAM["carve"])
    fit = H.sample_bags(world, int(W["n_fit"]), fit_mask, E.STREAM["fit"], **kw)
    val = H.sample_bags(world, int(W["n_val"]), val_mask, E.STREAM["val"], **kw)
    for name, bags, allowed in (("fit", fit, fit_mask), ("val", val, val_mask)):
        ok = H.check_bags(world, bags, allowed, int(W["n_distractors"]))
        M(f"check.{name}.construction", all(ok.values()), detail=ok)
    return {"world": world, "fit": fit, "val": val, "fit_mask": fit_mask, "val_mask": val_mask}


def leaf_stats(tm, kind: str, P: dict, world: H.HaoWorld) -> dict:
    inc = E.pkg_included(tm)  # bool [clauses, teams, literals_per_leaf]
    C, teams, L = inc.shape
    neg = bool(P["append_negated"])
    n_feat = L // 2 if neg else L
    w = E.pkg_weights(tm)[0]
    pos = w > 0
    group = world.word_group[:n_feat] >= 0
    lits = inc.sum(axis=2)  # [C, teams]
    empty = lits == 0
    pos_group = inc[:, :, :n_feat][:, :, group].sum(axis=2)
    pos_distr = inc[:, :, :n_feat][:, :, ~group].sum(axis=2)
    neg_lits = inc[:, :, n_feat:].sum(axis=2) if neg else np.zeros_like(lits)
    neg_group = inc[:, :, n_feat:][:, :, group].sum(axis=2) if neg else np.zeros_like(lits)
    out = {}
    for tag, sel in (("posc", pos), ("negc", ~pos)):
        if not sel.any():
            continue
        ne = ~empty[sel]
        out[f"{tag}.empty_leaf_share"] = float(empty[sel].mean())
        out[f"{tag}.lits_per_nonempty_leaf"] = float(lits[sel][ne].mean()) if ne.any() else 0.0
        out[f"{tag}.pos_group_lits_per_leaf"] = float(pos_group[sel].mean())
        out[f"{tag}.pos_distractor_lits_per_leaf"] = float(pos_distr[sel].mean())
        out[f"{tag}.neg_lits_per_leaf"] = float(neg_lits[sel].mean())
        out[f"{tag}.neg_group_lits_per_leaf"] = float(neg_group[sel].mean())
        sets = [set(np.flatnonzero(inc[i, :, :n_feat].any(axis=0)).tolist()) for i in np.flatnonzero(sel)]
        out[f"{tag}.group_word_coverage"] = E.group_word_coverage(sets, world)
    if kind == "hier":
        A, K = int(P["alternatives"]), int(P["n_concepts"])
        groups = H.concept_groups(inc, A, K, neg)
        rec = [H.or_group_recovery(groups[i], world.groups)["mean_jaccard"] for i in np.flatnonzero(pos)]
        out["posc.recovery_mean"] = float(np.mean(rec)) if rec else 0.0
        out["posc.recovery_max"] = float(np.max(rec)) if rec else 0.0
        recn = [H.or_group_recovery(groups[i], world.groups)["mean_jaccard"] for i in np.flatnonzero(~pos)]
        out["negc.recovery_mean"] = float(np.mean(recn)) if recn else 0.0
    return out


def evaluate(tm, D: dict, T: int) -> dict:
    fit, val = D["fit"], D["val"]
    s_fit = np.asarray(tm.score(np.ascontiguousarray(fit.X, dtype=np.uint32)), dtype=np.float64)
    s_val = np.asarray(tm.score(np.ascontiguousarray(val.X, dtype=np.uint32)), dtype=np.float64)
    p_fit = (s_fit >= 0).astype(np.int64)
    p_val = (s_val >= 0).astype(np.int64)
    yf, yv = fit.y.astype(np.int64), val.y.astype(np.int64)
    return {"train_acc": float(np.mean(p_fit == yf)), "val_acc": float(np.mean(p_val == yv)),
            "train_pos_rate": float(p_fit.mean()), "val_pos_rate": float(p_val.mean()),
            "train_zero_sum_share": float(np.mean(s_fit == 0)),
            "train_sum_mean_pos": float(s_fit[yf == 1].mean()), "train_sum_mean_neg": float(s_fit[yf == 0].mean()),
            "train_saturated_share": float(np.mean(np.abs(s_fit) >= T)),
            "val_sum_mean_pos": float(s_val[yv == 1].mean()), "val_sum_mean_neg": float(s_val[yv == 0].mean())}


def run_config(htm, spec: dict, seed: int, D: dict, n_features: int) -> dict:
    kind = spec["kind"]
    P = {k: v for k, v in spec.items() if k not in ("name", "kind", "epochs", "eval_every", "wall_cap_s", "note")}
    epochs = int(spec["epochs"])
    every = int(spec.get("eval_every", 5))
    cap = float(spec.get("wall_cap_s", 1800))
    tm = E.pkg_make(htm, kind, n_features, P, seed)
    Xf = np.ascontiguousarray(D["fit"].X, dtype=np.uint32)
    yf = np.ascontiguousarray(D["fit"].y, dtype=np.uint32)
    n_clauses = int(P["clauses"]) if kind == "hier" else int(P["clauses"]) * int(P["alternatives"]) * int(P["n_concepts"])
    cell = {"name": spec["name"], "kind": kind, "params": P, "epochs_planned": epochs, "seed": seed,
            "n_clauses_package": n_clauses, "curve": []}
    t0 = time.time()
    fit_s = 0.0
    for e in range(epochs):
        tf = time.time()
        tm.fit(Xf, yf)
        fit_s += time.time() - tf
        last = e + 1 == epochs or time.time() - t0 > cap
        if (e + 1) % every == 0 or e == 0 or last:
            row = {"epoch": e + 1, "fit_s_cum": fit_s, **evaluate(tm, D, int(P["T"])), **leaf_stats(tm, kind, P, D["world"])}
            cell["curve"].append(row)
            log(f"  {spec['name']} ep {e + 1}: train {row['train_acc']:.4f} val {row['val_acc']:.4f} pos_rate {row['train_pos_rate']:.3f} "
                f"zero {row['train_zero_sum_share']:.3f} sum+ {row['train_sum_mean_pos']:.1f} sum- {row['train_sum_mean_neg']:.1f} "
                f"empty+ {row.get('posc.empty_leaf_share', 0):.3f} lits+ {row.get('posc.lits_per_nonempty_leaf', 0):.1f} "
                f"grp+ {row.get('posc.pos_group_lits_per_leaf', 0):.2f} dis+ {row.get('posc.pos_distractor_lits_per_leaf', 0):.2f} "
                f"neg+ {row.get('posc.neg_lits_per_leaf', 0):.1f} cov+ {row.get('posc.group_word_coverage', 0):.2f} "
                f"rec+ {row.get('posc.recovery_mean', -1):.2f} ({fit_s:.0f} s)")
        if last:
            cell["epochs_run"] = e + 1
            break
    final = cell["curve"][-1]
    best = max(cell["curve"], key=lambda r: (r["val_acc"], -r["epoch"]))
    cell.update({"final_train_acc": final["train_acc"], "final_val_acc": final["val_acc"], "best_val_acc": best["val_acc"],
                 "best_val_epoch": best["epoch"], "train_acc_at_best_val": best["train_acc"],
                 "max_train_acc": max(r["train_acc"] for r in cell["curve"]), "fit_s": fit_s, "wall_s": time.time() - t0,
                 "state_sha256": E.pkg_state_sha(tm)})
    del tm
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="comma separated configuration names (default: all)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    P = cfg["params"]
    M("config_file", args.config, sha256=hashing.sha256_file(args.config))
    seed = int(P["seed"])
    if seed < 100:
        raise SystemExit("tuning is exploratory only: seeds >= 100 (docs/13 section 3)")
    D = tuning_data(seed, P["world"])
    for name in ("fit", "val"):
        b = D[name]
        xs, ys = E.arr_sha(b.X), E.arr_sha(b.y)
        M(f"data.{name}.X_sha256", xs)
        M(f"data.{name}.y_sha256", ys)
        ref = P.get("reference_hashes", {}).get(name)
        if ref:
            M(f"check.data.{name}.equals_exploratory_run", bool(ref["X_sha256"] == xs and ref["y_sha256"] == ys),
              reference=P.get("reference_run"))
    M("data.fit_combos", int(D["fit_mask"].sum()))
    M("data.val_combos", int(D["val_mask"].sum()))
    M("data.test_combinations_used", False, note="test bags are never sampled or scored by this script")
    htm = E.pkg_module()
    M("package.provenance", E.pkg_provenance())
    M("package.grid_block", {"grid": list(E.GRID), "block": list(E.BLOCK)})
    flush()
    configs = P["configs"]
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in keep]
    n_features = int(P["world"]["vocab"])
    for spec in configs:
        log(f"config {spec['name']}: {json.dumps({k: v for k, v in spec.items() if k != 'note'})}")
        cell = run_config(htm, spec, seed, D, n_features)
        with open(out_dir() / "cells.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(cell, sort_keys=True, default=float) + "\n")
        for k in ("final_train_acc", "final_val_acc", "best_val_acc", "best_val_epoch", "train_acc_at_best_val",
                  "max_train_acc", "fit_s"):
            M(f"{spec['name']}.{k}", cell[k])
        log(f"config {spec['name']} done: final train {cell['final_train_acc']:.4f} val {cell['final_val_acc']:.4f}; best val "
            f"{cell['best_val_acc']:.4f} at epoch {cell['best_val_epoch']} (train {cell['train_acc_at_best_val']:.4f}); "
            f"{cell['wall_s']:.0f} s")
        flush()
    M("wall_s", time.time() - T0)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
