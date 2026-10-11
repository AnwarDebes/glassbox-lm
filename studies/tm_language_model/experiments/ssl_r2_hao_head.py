"""EXPLORATORY pilot (docs/12 F27): the hierarchical package at upstream commit 5b4a223 on the H-AO world of
docs/13 A7 (H17.5), seed 100 only. Runs under .venv-htm/bin/python (built by env/htm_install.sh); .venv-tm and its
installed c85ca4e7 are never used here.

Why: on c85ca4e7 no configuration of 78 trained on H-AO (experiments/ssl_r2_hao_tune.py) and
the hierarchical vote saturates the class sum. HEAD 5b4a223 adds four constructor switches, read from its source
(third_party/ref/PyHierarchicalTsetlinMachineCUDA at 5b4a223, kernels.py and tm.py; read, never imported):
  binary_inference         evaluate_final adds the clause WEIGHT for a firing clause instead of weight x clause
                           output (the product over AND nodes of the number of firing OR alternatives); the class
                           sum used for training is the same sum, so the vote no longer saturates at +-T
  constant_update_p        update probability 1 for TA and weight feedback instead of |y - clip(class_sum)|/(2T)
  and_group_normalization  an AND node outputs the geometric mean of its children instead of their product
  no_clipping              score() returns the unclipped class sum; training still clips at +-T (update_hierarchy)
Derived no-ops, checked bit-exactly by the `expect_identical_to` configurations of round 0:
  no_clipping never changes training (kernel code is unaffected) and never changes a prediction (clipping keeps the
  sign), so it cannot change accuracy; and_group_normalization changes nothing when binary_inference is on (only
  "clause output > 0" is read, and powf(0, 1/n) = 0 keeps the zero pattern that the propagation reads); for the flat
  machine ((AND_GROUP, F), (AND_GROUP, 1)) a clause output is already 0 or 1 and the AND node has one child, so
  binary_inference and and_group_normalization are no-ops there too: only constant_update_p acts on the flat machine.

Stages (every stage is exploratory; the held-out TEST combinations of H-AO are never sampled or scored):
  --stage controls   positive controls with HEAD: (a) the AND-of-XORs demo (examples/ANDOfXORsDemo.py: its data
                     generator ported with a seeded RandomState, its argparse defaults: 4 ANDs, 1 irrelevant feature
                     per AND, 100,000 training and 1,000 test examples, 1 percent training label noise, 2 x 10 clauses
                     (number_of_clauses x alternatives, as the demo builds the hierarchical machine), T 40, s 2.5,
                     8 state bits, boost 0, 10 alternatives, hierarchy ((AND_GROUP, 3), (OR_ALTERNATIVES, 10),
                     (AND_GROUP, 4)); runs = 1 instead of 10) with the switch settings listed in the config;
                     (b) the noisy-parity demo (examples/NoisyParityDemo.py defaults: 32 clauses, T 250, s 25, 7 state
                     bits, boost 0, hierarchy ((AND_GROUP,3),(OR_ALTERNATIVES,3),(AND_GROUP,2),(OR_ALTERNATIVES,3),
                     (AND_GROUP,2))) on the SAME data and seed as the c85ca4e7 control
                     results/20260924T221350Z_ssl_r2_hao_control_exploratory (sha256 checked), for a reduced number of
                     epochs; with the switches off HEAD should reproduce that run's curve exactly (docs/12 F26 (3):
                     "by source diff behaves like HEAD 5b4a223 with HEAD's four new switches off"), which is checked.
  --stage tune       H-AO fit/validation tuning (the data of experiments/ssl_r2_hao_tune.py, sha256 checked against
                     the exploratory run), one configuration after another, per evaluation epoch: training and
                     validation accuracy, the diagnostics of ssl_r2_hao_tune.leaf_stats, and the OR-group read-outs:
                       R_node      (hier; the read-out of ssl_r2_hao.py) per USED positive-weight clause, the best
                                   one-to-one matching of its concept nodes' positive-literal word sets to the 4 true
                                   groups, mean Jaccard; mean, max and share >= 0.5 over those clauses
                       R_pos_pool  (both kinds) the best one-to-one matching of the 4 true groups to ALL positive-literal
                                   word sets of used positive-weight clauses (hier: per concept node; flat: per clause),
                                   mean Jaccard over the 4 groups
                       R_dm_pool   (both kinds, negated literals only) the same over the NEGATED-literal word sets of used
                                   negative-weight clauses (De Morgan: "no word of group g" is the complement of the
                                   OR-group g, the form in which a flat machine can hold an OR-group)
                     Optional per configuration: `dead_after` / `dead_below` (stop when the best training accuracy so
                     far is below dead_below at an evaluation at or after epoch dead_after), `stop_at_train` (stop once
                     training accuracy reaches it), `wall_cap_s`, `expect_identical_to` (bit-exact check of the TA
                     state and weights, and of the whole curve, against an earlier configuration of the same
                     invocation), `compare_to` (curve check against a configuration of an earlier c85ca4e7 run).

Launch (detached, CPU slot FIRST, docs/12 F25; from the study folder):
    setsid nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-htm/bin/python scripts/mem_guard.py \\
        --scope group --limit-gib 8 -- .venv-tm/bin/python scripts/run_logged.py --name ssl_r2_hao_head_<x>_exploratory \\
        --config <config> -- .venv-htm/bin/python -u experiments/ssl_r2_hao_head.py --stage <stage> --config <config> \\
        [--only a,b] [--kind hier|flat] > /dev/null 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import inspect
import io
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

import ssl_r2_hao as E  # noqa: E402  (grid/block, streams, include decoding, hierarchy shapes, provenance)
import ssl_r2_hao_tune as TU  # noqa: E402  (tuning data, evaluate, leaf_stats)

PIN = "5b4a223075c1e621577d508bc7c3c061230e29d5"
SWITCHES = ("binary_inference", "constant_update_p", "and_group_normalization", "no_clipping")
T0 = time.time()
METRICS: dict = {}
FAILED: list = []


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}


def check(name: str, ok: bool, **extra) -> None:
    M(f"check.{name}", bool(ok), **extra)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {extra}")


def flush() -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_metrics(METRICS, on_conflict="replace")


def out_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path("/tmp") / f"ssl_r2_hao_head_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def append_jsonl(name: str, row: dict) -> None:
    with open(out_dir() / name, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=float) + "\n")


# ------------------------------------------------------------------ the package at HEAD
def head_module():
    htm = E.pkg_module()
    prov = E.pkg_provenance()
    M("package.provenance", prov)
    commit = ((prov.get("direct_url") or {}).get("vcs_info") or {}).get("commit_id")
    sig = str(inspect.signature(htm.TsetlinMachine.__init__))
    M("package.TsetlinMachine_signature", sig)
    if commit != PIN or not all(s in sig for s in SWITCHES):
        raise SystemExit(f"expected PyHierarchicalTsetlinMachineCUDA at {PIN} with {SWITCHES}; found {commit}: run under .venv-htm")
    M("package.grid_block", {"grid": list(E.GRID), "block": list(E.BLOCK)})
    return htm


def switch_kwargs(P: dict) -> dict:
    return {"binary_inference": int(bool(P.get("binary_inference", 0))),
            "constant_update_p": int(bool(P.get("constant_update_p", 0))),
            "and_group_normalization": int(bool(P.get("and_group_normalization", 0))),
            "no_clipping": bool(P.get("no_clipping", 0))}


def pkg_make_head(htm, kind: str, n_features: int, P: dict, seed: int):
    """E.pkg_make with HEAD's four switches (hier: 'clauses' clauses; flat: clauses x alternatives x n_concepts)."""
    clauses = int(P["clauses"]) if kind == "hier" else int(P["clauses"]) * int(P.get("alternatives", 1)) * int(P.get("n_concepts", 1))
    return htm.TsetlinMachine(clauses, int(P["T"]), float(P["s"]), weighted_clauses=bool(P.get("weighted_clauses", False)),
                              hierarchy_structure=E.pkg_hierarchy(htm, kind, n_features, P),
                              boost_true_positive_feedback=int(P.get("boost", 0)),
                              number_of_state_bits=int(P.get("state_bits", 8)), append_negated=bool(P["append_negated"]),
                              grid=E.GRID, block=E.BLOCK, seed=int(seed), **switch_kwargs(P))


# ------------------------------------------------------------------ OR-group read-outs
def or_readouts(tm, kind: str, P: dict, world: H.HaoWorld) -> dict:
    inc = E.pkg_included(tm)
    C, teams, L = inc.shape
    neg = bool(P["append_negated"])
    n_feat = L // 2 if neg else L
    w = E.pkg_weights(tm)[0]
    pos = w > 0
    used = inc.reshape(C, -1).any(axis=1)
    out = {"clauses_used": int(used.sum()), "pos_clauses_used": int((used & pos).sum()),
           "neg_clauses_used": int((used & ~pos).sum()),
           "lits_per_used_clause": float(inc[used].reshape(int(used.sum()), -1).sum(axis=1).mean()) if used.any() else 0.0}
    pos_sets: list[set] = []
    if kind == "hier":
        A, K = int(P["alternatives"]), int(P["n_concepts"])
        groups = H.concept_groups(inc, A, K, neg)
        rec = [H.or_group_recovery(groups[i], world.groups)["mean_jaccard"] for i in np.flatnonzero(used & pos)]
        out["R_node_mean"] = float(np.mean(rec)) if rec else 0.0
        out["R_node_max"] = float(np.max(rec)) if rec else 0.0
        out["R_node_share_ge_0_5"] = float(np.mean(np.asarray(rec) >= 0.5)) if rec else 0.0
        for i in np.flatnonzero(used & pos):
            pos_sets.extend(g for g in groups[i] if g)
    else:
        for i in np.flatnonzero(used & pos):
            s = set(np.flatnonzero(inc[i, :, :n_feat].any(axis=0)).tolist())
            if s:
                pos_sets.append(s)
    out["R_pos_pool"] = H.or_group_recovery(pos_sets, world.groups)["mean_jaccard"] if pos_sets else 0.0
    if neg:
        dm_sets = []
        for i in np.flatnonzero(used & ~pos):
            s = set(np.flatnonzero(inc[i, :, n_feat:].any(axis=0)).tolist())
            if s:
                dm_sets.append(s)
        r = H.or_group_recovery(dm_sets, world.groups) if dm_sets else {"mean_jaccard": 0.0, "per_group": [0.0] * world.n_groups}
        out["R_dm_pool"] = r["mean_jaccard"]
        out["R_dm_pool_per_group"] = r["per_group"]
    return out


# ------------------------------------------------------------------ H-AO tuning
def compare_curves(a: list[dict], b: list[dict], fields: list[str], epochs: list[int] | None = None) -> tuple[bool, list]:
    ra = {r["epoch"]: r for r in a}
    rb = {r["epoch"]: r for r in b}
    eps = sorted(set(ra) & set(rb)) if epochs is None else epochs
    diffs = []
    for e in eps:
        if e not in ra or e not in rb:
            diffs.append({"epoch": e, "missing": True})
            continue
        for f in fields:
            if f in ra[e] and f in rb[e] and ra[e][f] != rb[e][f]:
                diffs.append({"epoch": e, "field": f, "a": ra[e][f], "b": rb[e][f]})
    return (len(diffs) == 0 and len(eps) > 0), diffs


CURVE_FIELDS = ["train_acc", "val_acc", "train_pos_rate", "val_pos_rate", "train_zero_sum_share", "train_sum_mean_pos",
                "train_sum_mean_neg", "posc.empty_leaf_share", "posc.lits_per_nonempty_leaf", "posc.pos_group_lits_per_leaf",
                "posc.pos_distractor_lits_per_leaf", "negc.empty_leaf_share"]


def run_config_head(htm, spec: dict, seed: int, D: dict, n_features: int, done: dict) -> dict:
    kind = spec["kind"]
    meta = ("name", "kind", "epochs", "eval_every", "wall_cap_s", "note", "dead_after", "dead_below", "stop_at_train",
            "expect_identical_to", "compare_to", "save_hierarchy", "counts_as_tuning")
    P = {k: v for k, v in spec.items() if k not in meta}
    epochs = int(spec["epochs"])
    every = int(spec.get("eval_every", 5))
    cap = float(spec.get("wall_cap_s", 1800))
    dead_after, dead_below = spec.get("dead_after"), spec.get("dead_below")
    stop_at = float(spec.get("stop_at_train", 2.0))
    tm = pkg_make_head(htm, kind, n_features, P, seed)
    Xf = np.ascontiguousarray(D["fit"].X, dtype=np.uint32)
    yf = np.ascontiguousarray(D["fit"].y, dtype=np.uint32)
    n_clauses = int(P["clauses"]) if kind == "hier" else int(P["clauses"]) * int(P.get("alternatives", 1)) * int(P.get("n_concepts", 1))
    cell = {"name": spec["name"], "kind": kind, "params": P, "epochs_planned": epochs, "seed": seed,
            "counts_as_tuning": bool(spec.get("counts_as_tuning", True)), "n_clauses_package": n_clauses,
            "n_leaves_total": n_clauses * (int(P["alternatives"]) * int(P["n_concepts"]) if kind == "hier" else 1),
            "curve": [], "stop_reason": "epochs"}
    t0 = time.time()
    fit_s = 0.0
    best_train = 0.0
    for e in range(epochs):
        tf = time.time()
        tm.fit(Xf, yf)
        fit_s += time.time() - tf
        last = e + 1 == epochs
        if time.time() - t0 > cap:
            last, cell["stop_reason"] = True, "wall_cap"
        if (e + 1) % every == 0 or e == 0 or last:
            row = {"epoch": e + 1, "fit_s_cum": fit_s, **TU.evaluate(tm, D, int(P["T"])),
                   **TU.leaf_stats(tm, kind, P, D["world"]), **or_readouts(tm, kind, P, D["world"])}
            cell["curve"].append(row)
            best_train = max(best_train, row["train_acc"])
            log(f"  {spec['name']} ep {e + 1}: train {row['train_acc']:.4f} val {row['val_acc']:.4f} pos_rate {row['train_pos_rate']:.3f} "
                f"sum+ {row['train_sum_mean_pos']:.1f} sum- {row['train_sum_mean_neg']:.1f} sat {row['train_saturated_share']:.2f} "
                f"empty+ {row.get('posc.empty_leaf_share', 0):.3f} grp+ {row.get('posc.pos_group_lits_per_leaf', 0):.2f} "
                f"dis+ {row.get('posc.pos_distractor_lits_per_leaf', 0):.2f} neg+ {row.get('posc.neg_lits_per_leaf', 0):.1f} "
                f"negneg {row.get('negc.neg_lits_per_leaf', 0):.1f} cov+ {row.get('posc.group_word_coverage', 0):.2f} "
                f"Rnode {row.get('R_node_mean', -1):.2f}/{row.get('R_node_max', -1):.2f} Rpos {row['R_pos_pool']:.2f} "
                f"Rdm {row.get('R_dm_pool', -1):.2f} used {row['pos_clauses_used']}+/{row['neg_clauses_used']}- ({fit_s:.0f} s)")
            if not last and row["train_acc"] >= stop_at:
                last, cell["stop_reason"] = True, f"train_acc >= {stop_at}"
            if not last and dead_after is not None and e + 1 >= int(dead_after) and best_train < float(dead_below):
                last, cell["stop_reason"] = True, f"dead: best train {best_train:.4f} < {dead_below} at epoch {e + 1}"
        if last:
            cell["epochs_run"] = e + 1
            break
    final = cell["curve"][-1]
    best = max(cell["curve"], key=lambda r: (r["val_acc"], -r["epoch"]))
    cell.update({"final_train_acc": final["train_acc"], "final_val_acc": final["val_acc"], "best_val_acc": best["val_acc"],
                 "best_val_epoch": best["epoch"], "train_acc_at_best_val": best["train_acc"],
                 "max_train_acc": max(r["train_acc"] for r in cell["curve"]), "fit_s": fit_s, "wall_s": time.time() - t0,
                 "state_sha256": E.pkg_state_sha(tm)})
    for k in ("R_node_mean", "R_node_max", "R_node_share_ge_0_5", "R_pos_pool", "R_dm_pool"):
        if k in final:
            cell[f"final_{k}"] = final[k]
            cell[f"{k}_at_best_val"] = best.get(k)
    if kind == "hier" and (spec.get("save_hierarchy") or final["train_acc"] >= 0.9):
        inc = E.pkg_included(tm)
        hier = [tuple(x) for x in E.pkg_hierarchy(htm, kind, n_features, P)]
        text = H.render_hierarchy(inc, hier, bool(P["append_negated"]))
        fn = out_dir() / f"{spec['name']}.hierarchy.txt.gz"
        with gzip.open(fn, "wt", encoding="utf-8") as fh:
            fh.write(text)
        cell["hierarchy_file"] = fn.name
    ref = spec.get("expect_identical_to")
    if ref:
        other = done.get(ref)
        if other is None:
            check(f"{spec['name']}.identical_to_{ref}", False, reason="reference configuration not run in this invocation")
        else:
            same_state = other["state_sha256"] == cell["state_sha256"]
            same_curve, diffs = compare_curves(other["curve"], cell["curve"], ["train_acc", "val_acc", "train_pos_rate", "val_pos_rate"])
            check(f"{spec['name']}.identical_to_{ref}", same_state and same_curve, state_equal=same_state,
                  accuracy_curve_equal=same_curve, diffs=diffs[:10])
            cell["identical_to"] = {"ref": ref, "state_equal": same_state, "accuracy_curve_equal": same_curve}
    cmp = spec.get("compare_to")
    if cmp:
        ref_cell = None
        with open(ROOT / cmp["run"] / "cells.jsonl", encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                if c["name"] == cmp["name"]:
                    ref_cell = c
        if ref_cell is None:
            check(f"{spec['name']}.equals_c85_{cmp['name']}", False, reason="reference cell not found")
        else:
            ok, diffs = compare_curves(ref_cell["curve"], cell["curve"], CURVE_FIELDS, [int(x) for x in cmp["epochs"]])
            check(f"{spec['name']}.equals_c85_{cmp['name']}", ok, reference_run=cmp["run"], epochs=cmp["epochs"],
                  fields=CURVE_FIELDS, diffs=diffs[:10])
            cell["equals_c85"] = {"ref": cmp, "equal": ok}
    del tm
    return cell


def stage_tune(cfg: dict, args) -> int:
    P = cfg["params"]
    seed = int(P["seed"])
    if seed < 100:
        raise SystemExit("exploratory only: seeds >= 100 (docs/13 section 3)")
    D = TU.tuning_data(seed, P["world"])
    METRICS.update(TU.METRICS)
    for name in ("fit", "val"):
        b = D[name]
        xs, ys = E.arr_sha(b.X), E.arr_sha(b.y)
        M(f"data.{name}.X_sha256", xs)
        M(f"data.{name}.y_sha256", ys)
        ref = P.get("reference_hashes", {}).get(name)
        if ref:
            check(f"data.{name}.equals_exploratory_run", ref["X_sha256"] == xs and ref["y_sha256"] == ys,
                  reference=P.get("reference_run"))
    M("data.fit_combos", int(D["fit_mask"].sum()))
    M("data.val_combos", int(D["val_mask"].sum()))
    M("data.world_groups", D["world"].groups.tolist())
    M("data.test_combinations_used", False, note="test bags are never sampled or scored by this script")
    htm = head_module()
    flush()
    configs = P["configs"]
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in keep]
    if args.kind:
        configs = [c for c in configs if c["kind"] == args.kind]
    n_features = int(P["world"]["vocab"])
    done: dict = {}
    for spec in configs:
        log(f"config {spec['name']}: {json.dumps({k: v for k, v in spec.items() if k != 'note'})}")
        cell = run_config_head(htm, spec, seed, D, n_features, done)
        done[spec["name"]] = cell
        append_jsonl("cells.jsonl", cell)
        for k in ("final_train_acc", "final_val_acc", "best_val_acc", "best_val_epoch", "train_acc_at_best_val",
                  "max_train_acc", "fit_s", "epochs_run", "stop_reason", "final_R_node_max", "final_R_pos_pool", "final_R_dm_pool"):
            if k in cell:
                M(f"{spec['name']}.{k}", cell[k])
        log(f"config {spec['name']} done ({cell['stop_reason']}): final train {cell['final_train_acc']:.4f} val {cell['final_val_acc']:.4f}; "
            f"best val {cell['best_val_acc']:.4f} at epoch {cell['best_val_epoch']} (train {cell['train_acc_at_best_val']:.4f}); "
            f"max train {cell['max_train_acc']:.4f}; {cell['wall_s']:.0f} s")
        flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ positive controls
def andxor_data(seed: int, n_ands: int, irr: int, n_train: int, n_test: int, noise: float):
    """examples/ANDOfXORsDemo.py's generator, statement for statement, with np.random replaced by one seeded
    RandomState (the demo draws from the unseeded global generator)."""
    rs = np.random.RandomState(seed)
    w = 2 + irr
    and_factors = np.empty(n_ands, dtype=np.uint32)

    def block(n):
        X = rs.randint(2, size=(n, n_ands * w), dtype=np.uint32)
        Y = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            Y[i] = rs.randint(2)
            if Y[i] == 1:
                and_factors[:] = 1
                for j in range(n_ands):
                    if rs.random_sample() <= 0.5:
                        X[i, j * w:j * w + 2] = [0, 1]
                    else:
                        X[i, j * w:j * w + 2] = [1, 0]
            else:
                number_of_true_factors = rs.randint(n_ands)
                and_factors[:] = 0
                and_factors[:number_of_true_factors] = 1
                rs.shuffle(and_factors)
            for j in range(n_ands):
                if and_factors[j]:
                    if rs.random_sample() <= 0.5:
                        X[i, j * w:j * w + 2] = [0, 1]
                    else:
                        X[i, j * w:j * w + 2] = [1, 0]
                else:
                    if rs.random_sample() <= 0.5:
                        X[i, j * w:j * w + 2] = [0, 0]
                    else:
                        X[i, j * w:j * w + 2] = [1, 1]
        return X, Y

    Xtr, Ytr = block(n_train)
    Ytr_noisy = np.where(rs.rand(n_train) <= noise, 1 - Ytr, Ytr).astype(np.uint32)
    Xte, Yte = block(n_test)
    return Xtr, Ytr_noisy, Ytr, Xte, Yte


def xor_rule(X: np.ndarray, n_ands: int, irr: int) -> np.ndarray:
    w = 2 + irr
    return np.all(np.stack([X[:, j * w] != X[:, j * w + 1] for j in range(n_ands)], axis=1), axis=1).astype(np.uint32)


def control_andxor(htm, C: dict) -> None:
    Xtr, ytr, ytr_clean, Xte, yte = andxor_data(int(C["data_seed"]), int(C["n_ands"]), int(C["irrelevant"]),
                                                int(C["n_train"]), int(C["n_test"]), float(C["noise"]))
    check("andxor.labels_follow_rule", bool(np.array_equal(xor_rule(Xtr, int(C["n_ands"]), int(C["irrelevant"])), ytr_clean))
          and bool(np.array_equal(xor_rule(Xte, int(C["n_ands"]), int(C["irrelevant"])), yte)))
    M("andxor.data", {"Xtr": E.arr_sha(Xtr), "ytr": E.arr_sha(ytr), "Xte": E.arr_sha(Xte), "yte": E.arr_sha(yte),
                      "train_noise_rate": float(np.mean(ytr != ytr_clean)), "train_pos_rate": float(ytr_clean.mean()),
                      "test_pos_rate": float(yte.mean())})
    w = 2 + int(C["irrelevant"])
    for v in C["variants"]:
        sw = switch_kwargs(v)
        A = int(C["alternatives"])
        if v.get("vanilla"):
            tm = htm.TsetlinMachine(int(C["clauses"]), int(C["T"]), float(C["s"]), number_of_state_bits=int(C["state_bits"]),
                                    boost_true_positive_feedback=0, hierarchy_structure=((htm.AND_GROUP, w * int(C["n_ands"])), (htm.OR_ALTERNATIVES, A)),
                                    grid=E.GRID, block=E.BLOCK, seed=int(C["tm_seed"]), **sw)
        else:
            tm = htm.TsetlinMachine(int(C["clauses"]) * A, int(C["T"]), float(C["s"]), number_of_state_bits=int(C["state_bits"]),
                                    boost_true_positive_feedback=0,
                                    hierarchy_structure=((htm.AND_GROUP, w), (htm.OR_ALTERNATIVES, A), (htm.AND_GROUP, int(C["n_ands"]))),
                                    grid=E.GRID, block=E.BLOCK, seed=int(C["tm_seed"]), **sw)
        curve = []
        t0 = time.time()
        for e in range(int(C["epochs"])):
            te = time.time()
            tm.fit(Xtr, ytr)
            ep_s = time.time() - te
            acc = float(np.mean(np.asarray(tm.predict(Xte)) == yte))
            curve.append({"epoch": e + 1, "test_acc": acc, "epoch_s": ep_s})
            log(f"andxor {v['name']} epoch {e + 1}: test {acc:.4f} ({ep_s:.1f} s)")
            if time.time() - t0 > float(C["wall_cap_s"]):
                log(f"andxor {v['name']}: wall cap")
                break
        best = max(r["test_acc"] for r in curve)
        M(f"andxor.{v['name']}.curve", curve)
        M(f"andxor.{v['name']}.final_test_acc", curve[-1]["test_acc"])
        M(f"andxor.{v['name']}.best_test_acc", best)
        M(f"andxor.{v['name']}.learns", best >= float(C["learn_threshold"]),
          note=f"learns = best clean test accuracy >= {C['learn_threshold']} (exploratory reading, not a gate)")
        M(f"andxor.{v['name']}.switches", {**sw, "vanilla": bool(v.get("vanilla", False))})
        append_jsonl("controls.jsonl", {"control": "andxor", "variant": v, "curve": curve, "state_sha256": E.pkg_state_sha(tm)})
        flush()
        del tm


def control_parity(htm, C: dict) -> None:
    seed = int(C["seed"])
    Xtr, ytr, ytr_clean = H.noisy_parity(int(C["n_train"]), seed, E.STREAM["parity_train"], float(C["noise"]))
    Xte, _, yte = H.noisy_parity(int(C["n_test"]), seed, E.STREAM["parity_test"], 0.0)
    shas = {"Xtr": E.arr_sha(Xtr), "ytr": E.arr_sha(ytr), "Xte": E.arr_sha(Xte), "yte": E.arr_sha(yte)}
    M("parity.data_sha256", shas)
    ref_run = ROOT / C["reference_run"]
    ref = json.loads((ref_run / "metrics.json").read_text(encoding="utf-8"))
    check("parity.data_equals_c85_control", ref["control.data_sha256"]["value"] == shas, reference_run=C["reference_run"])
    ref_curve = ref["control.curve"]["value"]
    hier = tuple((getattr(htm, t), int(n)) for t, n in C["hierarchy"])
    for v in C["variants"]:
        sw = switch_kwargs(v)
        tm = htm.TsetlinMachine(int(C["clauses"]), int(C["T"]), float(C["s"]), number_of_state_bits=int(C["state_bits"]),
                                boost_true_positive_feedback=int(C["boost"]), hierarchy_structure=hier,
                                grid=E.GRID, block=E.BLOCK, seed=seed, **sw)
        curve = []
        t0 = time.time()
        for e in range(int(v["epochs"])):
            te = time.time()
            tm.fit(Xtr, ytr)
            ep_s = time.time() - te
            if (e + 1) % int(C["eval_every"]) == 0 or e == 0 or e + 1 == int(v["epochs"]):
                acc = float(np.mean(np.asarray(tm.predict(Xte)) == yte))
                curve.append({"epoch": e + 1, "test_acc_clean": acc, "epoch_s": ep_s, "elapsed_s": time.time() - t0})
                log(f"parity {v['name']} epoch {e + 1}: clean test {acc:.4f} ({ep_s:.1f} s)")
                M(f"parity.{v['name']}.curve", curve)
                flush()
            if time.time() - t0 > float(C["wall_cap_s"]):
                log(f"parity {v['name']}: wall cap")
                break
        M(f"parity.{v['name']}.switches", sw)
        M(f"parity.{v['name']}.best_test_acc_clean", max(r["test_acc_clean"] for r in curve))
        if v.get("compare_to_c85"):
            ra = {r["epoch"]: r["test_acc_clean"] for r in ref_curve}
            pairs = [(r["epoch"], r["test_acc_clean"], ra.get(r["epoch"])) for r in curve if r["epoch"] in ra]
            check(f"parity.{v['name']}.curve_equals_c85_control", len(pairs) > 0 and all(a == b for _, a, b in pairs),
                  pairs=pairs, reference_run=C["reference_run"])
        if v.get("check_print"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tm.print_hierarchy()
            (out_dir() / f"parity_{v['name']}_hierarchy.txt").write_text(buf.getvalue(), encoding="utf-8")
            inc = E.pkg_included(tm)
            check(f"parity.{v['name']}.render_equals_head_print_hierarchy",
                  H.render_hierarchy(inc, [tuple(x) for x in hier], True) == buf.getvalue())
        append_jsonl("controls.jsonl", {"control": "parity", "variant": v, "curve": curve, "state_sha256": E.pkg_state_sha(tm)})
        flush()
        del tm


def stage_controls(cfg: dict, args) -> int:
    htm = head_module()
    flush()
    which = set((args.only or "andxor,parity").split(","))
    if "andxor" in which:
        control_andxor(htm, cfg["params"]["andxor"])
    if "parity" in which:
        control_parity(htm, cfg["params"]["parity"])
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["controls", "tune"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="tune: configuration names; controls: andxor,parity")
    ap.add_argument("--kind", default=None, choices=["hier", "flat"], help="tune: only configurations of this kind")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    M("config_file", args.config, sha256=hashing.sha256_file(args.config))
    M("stage", args.stage)
    M("label", "EXPLORATORY (docs/12 F27; seed 100; not a result, docs/13 section 3)")
    if args.stage == "controls":
        return stage_controls(cfg, args)
    return stage_tune(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
