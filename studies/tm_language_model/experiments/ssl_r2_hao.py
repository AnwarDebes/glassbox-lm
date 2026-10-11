"""R2 mechanism test H-AO (docs/13 amendment A7, hypothesis H17.5): is Granmo's AND-OR hierarchy NEEDED to learn
an AND of OR-groups from bags of words, and does the package recover the OR-groups it learned?

World: src/tmlm/ssl/hao_world.py (vocabulary 2,000; four OR-groups of 6 words; bags with 24 distractors; positives
hold at least one word of every group, negatives the same number of group words with one group missing; training
bags use a random 30 percent of the 1,296 one-word-per-group combinations, test bags the other 70 percent).

Models (all on the same binary bag-of-words matrices of one world seed):
  htm_hier[_neg|_pos]  PyHierarchicalTsetlinMachineCUDA.tm.TsetlinMachine (the installed git c85ca4e7, docs/02 V3),
                       hierarchy ((AND_GROUP, 2000), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, 4)): a clause is an
                       AND of 4 concept nodes, each an OR of A leaf conjunctions over all 2,000 features (the shape
                       of Granmo's MultiConceptDemo and of the registered H1 config)
  htm_flat[_neg|_pos]  the same class, hierarchy ((AND_GROUP, 2000), (AND_GROUP, 1)), clauses x A x 4 clauses (equal
                       leaf count, the package demos' --vanilla convention)
                       _neg: append_negated=True (the package default; PRIMARY pair for H17.5, fixed before any
                       output: the hierarchy is only "needed" if the flat machine fails WITH negated literals, which
                       can express "group g missing" as one conjunction of 6 negations); _pos: append_negated=False
                       (Granmo's bag-task demos and the H1 config; SECONDARY pair)
  tmu_tm, tmu_ctm      tmu TMClassifier and TMCoalescedClassifier, CPU, feature_negation=True (negated literals)
  lr                   L2 logistic regression, C by 4-fold CV inside the training bags (tmlm.ssl.transfer.lr_cell)
  gbt                  sklearn HistGradientBoostingClassifier (no early stopping, random_state = seed)
  mlp                  sklearn MLPClassifier, one hidden layer (random_state = seed)
Hyperparameters: a model whose config lists a grid of more than one point is selected on VALIDATION bags built from
validation combinations carved from the training combinations (20 percent, fit bags from the other 80 percent),
then refit on all training bags; the test bags (the 70 percent test combinations) are scored once per seed.

Metrics per (seed, model): held-out (test) accuracy, training accuracy, clauses used (clauses with at least one
included literal), for htm_hier the OR-group recovery (per positive-polarity used clause, the Jaccard of its 4
concept nodes' positive-literal word sets against the 4 true groups under the best matching; mean, max, share with
recovery >= 0.5; read from the package's printed hierarchy, which is also saved), for every Tsetlin model the
group-word coverage (share of the 24 group words included as a positive literal in some used clause that votes for
the positive class), wall times.

Positive control (--stage control): Granmo's noisy-parity demo exactly as the package ships it
(examples/NoisyParityDemo.py, NoisyParityData.py: 12 features, parity of 4 variables of 2 bits, 20 percent label
noise on the training labels, clean test labels; 200,000 training and 200,000 test examples; 32 clauses, T 250,
s 25.0, 7 state bits, no boost; hierarchy ((AND_GROUP,3),(OR_ALTERNATIVES,3),(AND_GROUP,2),(OR_ALTERNATIVES,3),
(AND_GROUP,2))), trained until the clean test accuracy reaches the config's pass threshold, the epoch cap or the
wall cap; the curve and print_hierarchy are logged. A failed control is reported, never hidden.

Verdict (--stage report): H17.5 on seeds 1 to 5: hierarchical mean test accuracy >= 0.90 AND >= flat + 0.20,
primary pair (_neg), secondary pair (_pos) reported next to it; paired per-seed differences with a Student-t
interval and the one-sided Wilcoxon signed-rank p (with 5 seeds the smallest reachable p is 1/32).

Stages and launch commands (detached, CPU slot FIRST, docs/12 F25; run from the study folder after
`source env/env.sh`; the GPU stages need the shared GPU lock, the CPU stage does not):

  confirmatory (seeds 1 to 5), config experiments/cvtm/configs/ssl_r2_hao.json:
    nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/mem_guard.py \\
        --scope group --limit-gib 8 -- scripts/run_logged.py --name ssl_r2_hao_gpu \\
        --config experiments/cvtm/configs/ssl_r2_hao.json -- .venv-tm/bin/python -u experiments/ssl_r2_hao.py \\
        --stage gpu --config experiments/cvtm/configs/ssl_r2_hao.json --replay-check > /dev/null 2>&1 &
    nohup scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 8 -- \\
        scripts/run_logged.py --no-gpu-sampler --name ssl_r2_hao_cpu --config experiments/cvtm/configs/ssl_r2_hao.json \\
        -- .venv-tm/bin/python -u experiments/ssl_r2_hao.py --stage cpu --config experiments/cvtm/configs/ssl_r2_hao.json \\
        --replay-check > /dev/null 2>&1 &
    nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/mem_guard.py \\
        --scope group --limit-gib 8 -- scripts/run_logged.py --name ssl_r2_hao_control \\
        --config experiments/cvtm/configs/ssl_r2_hao.json -- .venv-tm/bin/python -u experiments/ssl_r2_hao.py \\
        --stage control --config experiments/cvtm/configs/ssl_r2_hao.json > /dev/null 2>&1 &
    then (CPU, seconds):
    scripts/cpu_slot.sh 1 -- scripts/run_logged.py --no-gpu-sampler --name ssl_r2_hao_report \\
        --config experiments/cvtm/configs/ssl_r2_hao.json -- .venv-tm/bin/python -u experiments/ssl_r2_hao.py \\
        --stage report --config experiments/cvtm/configs/ssl_r2_hao.json --runs <gpu run dir>,<cpu run dir>,<control run dir>
  exploratory seed 100 first: the same commands with experiments/cvtm/configs/ssl_r2_hao_seed100.json and names
  ssl_r2_hao_*_exploratory. Smoke: experiments/cvtm/configs/ssl_r2_hao_smoke.json (1 seed, small models).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import hao_world as H  # noqa: E402

T0 = time.time()
METRICS: dict = {}
FAILED: list = []
GRID = (16 * 13, 1, 1)
BLOCK = (128, 1, 1)
STREAM = {"train": 1, "test": 2, "fit": 3, "val": 4, "carve": 5, "replay": 6, "parity_train": 11, "parity_test": 12}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}


def flush() -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_metrics(METRICS, on_conflict="replace")


def check(name: str, ok: bool, **extra) -> None:
    M(f"check.{name}", bool(ok), **extra)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {extra}")


def run_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_r2_hao_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def write_cell(cell: dict) -> None:
    with open(run_dir() / "cells.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(cell, sort_keys=True, default=float) + "\n")


def arr_sha(a: np.ndarray) -> str:
    return hashing.sha256_array(np.ascontiguousarray(a))


# ------------------------------------------------------------------ data of one seed
def world_data(seed: int, W: dict) -> dict:
    world = H.make_world(seed, vocab=int(W["vocab"]), n_groups=int(W["n_groups"]), group_size=int(W["group_size"]),
                         train_fraction=float(W["train_fraction"]))
    kw = {"n_distractors": int(W["n_distractors"]), "extra_probs": tuple(W["extra_probs"])}
    train = H.sample_bags(world, int(W["n_train"]), world.train_combo, STREAM["train"], **kw)
    test = H.sample_bags(world, int(W["n_test"]), ~world.train_combo, STREAM["test"], **kw)
    fit_mask, val_mask = H.carve_combos(world, world.train_combo, float(W["val_fraction"]), STREAM["carve"])
    fit = H.sample_bags(world, int(W["n_fit"]), fit_mask, STREAM["fit"], **kw)
    val = H.sample_bags(world, int(W["n_val"]), val_mask, STREAM["val"], **kw)
    for name, bags, allowed in (("train", train, world.train_combo), ("test", test, ~world.train_combo),
                                ("fit", fit, fit_mask), ("val", val, val_mask)):
        for k, ok in H.check_bags(world, bags, allowed, int(W["n_distractors"])).items():
            check(f"seed{seed}.{name}.{k}", ok)
    M(f"seed{seed}.world.groups", world.groups.tolist())
    M(f"seed{seed}.world.train_combos", int(world.train_combo.sum()))
    M(f"seed{seed}.world.fit_combos", int(fit_mask.sum()))
    M(f"seed{seed}.world.val_combos", int(val_mask.sum()))
    for name, b in (("train", train), ("test", test), ("fit", fit), ("val", val)):
        M(f"seed{seed}.data.{name}.n", b.n)
        M(f"seed{seed}.data.{name}.X_sha256", arr_sha(b.X))
        M(f"seed{seed}.data.{name}.y_sha256", arr_sha(b.y))
    return {"world": world, "train": train, "test": test, "fit": fit, "val": val}


# ------------------------------------------------------------------ the hierarchical package (GPU)
def pkg_module():
    import PyHierarchicalTsetlinMachineCUDA.tm as htm  # creates the CUDA context (pycuda.autoprimaryctx)

    return htm


def pkg_provenance() -> dict:
    import importlib.metadata as md

    import PyHierarchicalTsetlinMachineCUDA as pkg

    d = Path(pkg.__file__).parent
    dist = md.distribution("PyHierarchicalTsetlinMachineCUDA")
    direct = dist.read_text("direct_url.json")
    return {"version": dist.version, "direct_url": json.loads(direct) if direct else None,
            "tm_py_sha256": hashing.sha256_file(d / "tm.py"), "kernels_py_sha256": hashing.sha256_file(d / "kernels.py")}


def pkg_hierarchy(htm, kind: str, n_features: int, P: dict):
    if kind == "hier":
        return ((htm.AND_GROUP, n_features), (htm.OR_ALTERNATIVES, int(P["alternatives"])),
                (htm.AND_ALTERNATIVES, int(P["n_concepts"])))
    return ((htm.AND_GROUP, n_features), (htm.AND_GROUP, 1))


def pkg_make(htm, kind: str, n_features: int, P: dict, seed: int):
    clauses = int(P["clauses"]) if kind == "hier" else int(P["clauses"]) * int(P["alternatives"]) * int(P["n_concepts"])
    return htm.TsetlinMachine(clauses, int(P["T"]), float(P["s"]), weighted_clauses=bool(P.get("weighted_clauses", False)),
                              hierarchy_structure=pkg_hierarchy(htm, kind, n_features, P),
                              boost_true_positive_feedback=int(P.get("boost", 0)),
                              number_of_state_bits=int(P.get("state_bits", 8)), append_negated=bool(P["append_negated"]),
                              grid=GRID, block=BLOCK, seed=int(seed))


def pkg_included(tm) -> np.ndarray:
    """bool [clauses, ta_teams, literals_per_leaf]: TA action (include) bits from ONE device copy of the state
    (layout [clause][team][chunk][state bit], action = the top state bit, as tm.ta_action reads it)."""
    import pycuda.driver as cuda

    C, teams = int(tm.number_of_clauses), int(tm.number_of_ta_teams)
    ch, bits, L = int(tm.number_of_literal_chunks_per_leaf), int(tm.number_of_state_bits), int(tm.number_of_literals_per_leaf)
    raw = np.empty(C * teams * ch * bits, dtype=np.uint32)
    cuda.memcpy_dtoh(raw, tm.ta_state_hierarchy_gpu)
    act = np.ascontiguousarray(raw.reshape(C, teams, ch, bits)[:, :, :, bits - 1])
    b = np.unpackbits(act.view(np.uint8).reshape(C, teams, ch * 4), axis=2, bitorder="little")
    return b[:, :, :L].astype(bool)


def pkg_state_sha(tm) -> str:
    import pycuda.driver as cuda

    raw = np.empty(int(tm.number_of_clauses) * int(tm.number_of_ta_teams) * int(tm.number_of_literal_chunks_per_leaf)
                   * int(tm.number_of_state_bits), dtype=np.uint32)
    cuda.memcpy_dtoh(raw, tm.ta_state_hierarchy_gpu)
    w = np.empty(int(tm.number_of_outputs) * int(tm.number_of_clauses), dtype=np.int32)
    cuda.memcpy_dtoh(w, tm.clause_weights_gpu)
    return hashlib.sha256(raw.tobytes() + w.tobytes()).hexdigest()


def pkg_weights(tm) -> np.ndarray:
    import pycuda.driver as cuda

    w = np.empty(int(tm.number_of_outputs) * int(tm.number_of_clauses), dtype=np.int32)
    cuda.memcpy_dtoh(w, tm.clause_weights_gpu)
    return w.reshape(int(tm.number_of_outputs), int(tm.number_of_clauses))


def pkg_fit_predict(htm, kind: str, P: dict, seed: int, Xtr, ytr, Xev_list, epochs: int, on_epoch=None):
    tm = pkg_make(htm, kind, Xtr.shape[1], P, seed)
    Xtr32 = np.ascontiguousarray(Xtr, dtype=np.uint32)
    ytr32 = np.ascontiguousarray(ytr, dtype=np.uint32)
    t0 = time.time()
    for e in range(epochs):
        tm.fit(Xtr32, ytr32)
        if on_epoch is not None:
            on_epoch(e, tm)
    fit_s = time.time() - t0
    t0 = time.time()
    preds = [np.asarray(tm.predict(np.ascontiguousarray(X, dtype=np.uint32))).astype(np.int64) for X in Xev_list]
    return tm, preds, fit_s, time.time() - t0


def group_word_coverage(pos_literal_sets: list[set], world: H.HaoWorld) -> float:
    gw = {int(w) for w in world.groups.reshape(-1)}
    got = set().union(*pos_literal_sets) if pos_literal_sets else set()
    return len(gw & got) / len(gw)


def pkg_stats(tm, kind: str, P: dict, world: H.HaoWorld, out_prefix: Path | None) -> dict:
    inc = pkg_included(tm)
    L = inc.shape[2]
    neg = bool(P["append_negated"])
    n_pos_lit = L // 2 if neg else L
    used = inc.reshape(inc.shape[0], -1).any(axis=1)
    w = pkg_weights(tm)[0]
    positive = w > 0
    st = {"clauses": int(inc.shape[0]), "clauses_used": int(used.sum()), "positive_clauses_used": int((used & positive).sum()),
          "included_literals_mean_per_used_clause": float(inc[used].reshape(int(used.sum()), -1).sum(axis=1).mean()) if used.any() else 0.0}
    pos_sets = [set(np.flatnonzero(inc[i, :, :n_pos_lit].any(axis=0)).tolist()) for i in range(inc.shape[0]) if used[i] and positive[i]]
    st["group_word_coverage"] = group_word_coverage(pos_sets, world)
    if kind == "hier":
        hier = [tuple(x) for x in pkg_hierarchy(pkg_module(), kind, L // 2 if neg else L, P)]
        text = H.render_hierarchy(inc, hier, neg)
        arr_groups = H.concept_groups(inc, int(P["alternatives"]), int(P["n_concepts"]), neg)
        txt_groups = H.parse_printed_concepts(text, int(P["n_concepts"]))
        st["printed_groups_equal_array_groups"] = bool(txt_groups == arr_groups)
        rec = [H.or_group_recovery(txt_groups[i], world.groups)["mean_jaccard"] for i in range(inc.shape[0]) if used[i] and positive[i]]
        st["recovery_mean"] = float(np.mean(rec)) if rec else 0.0
        st["recovery_max"] = float(np.max(rec)) if rec else 0.0
        st["recovery_share_ge_0.5"] = float(np.mean(np.asarray(rec) >= 0.5)) if rec else 0.0
        st["recovery_per_positive_used_clause"] = rec
        if out_prefix is not None:
            out_prefix.with_suffix(".hierarchy.txt").write_text(text, encoding="utf-8")
            st["hierarchy_file"] = out_prefix.with_suffix(".hierarchy.txt").name
    return st


# ------------------------------------------------------------------ tmu (CPU)
def tmu_included(cb) -> np.ndarray:
    """bool [clauses, literals] from a tmu dense clause bank (flat uint32 [clause][chunk][state bit])."""
    C, ch, bits, L = int(cb.number_of_clauses), int(cb.number_of_ta_chunks), int(cb.number_of_state_bits_ta), int(cb.number_of_literals)
    act = np.ascontiguousarray(np.asarray(cb.clause_bank).reshape(C, ch, bits)[:, :, bits - 1])
    b = np.unpackbits(act.view(np.uint8).reshape(C, ch * 4), axis=1, bitorder="little")
    return b[:, :L].astype(bool)


def tmu_make(kind: str, P: dict, seed: int):
    np.random.seed(int(seed))
    if kind == "tmu_tm":
        from tmu.models.classification.vanilla_classifier import TMClassifier
        return TMClassifier(number_of_clauses=int(P["clauses"]), T=int(P["T"]), s=float(P["s"]), platform="CPU",
                            weighted_clauses=bool(P.get("weighted_clauses", False)), feature_negation=True, seed=int(seed))
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    return TMCoalescedClassifier(number_of_clauses=int(P["clauses"]), T=int(P["T"]), s=float(P["s"]), platform="CPU",
                                 weighted_clauses=True, feature_negation=True, seed=int(seed))


def tmu_fit_predict(kind: str, P: dict, seed: int, Xtr, ytr, Xev_list, epochs: int):
    tm = tmu_make(kind, P, seed)
    Xtr32 = np.ascontiguousarray(Xtr, dtype=np.uint32)
    ytr32 = np.ascontiguousarray(ytr, dtype=np.uint32)
    t0 = time.time()
    for _ in range(epochs):
        tm.fit(Xtr32, ytr32)
    fit_s = time.time() - t0
    t0 = time.time()
    preds = [np.asarray(tm.predict(np.ascontiguousarray(X, dtype=np.uint32))).astype(np.int64) for X in Xev_list]
    return tm, preds, fit_s, time.time() - t0


def tmu_state_sha(tm, kind: str) -> str:
    h = hashlib.sha256()
    banks = [tm.clause_bank] if kind == "tmu_ctm" else [tm.clause_banks[c] for c in range(len(tm.weight_banks))]
    for cb in banks:
        h.update(np.ascontiguousarray(cb.clause_bank).tobytes())
    for c in range(len(tm.weight_banks)):
        h.update(np.ascontiguousarray(tm.weight_banks[c].get_weights()).tobytes())
    return h.hexdigest()


def tmu_stats(tm, kind: str, world: H.HaoWorld) -> dict:
    if kind == "tmu_ctm":
        inc = tmu_included(tm.clause_bank)
        w1 = np.asarray(tm.weight_banks[1].get_weights())
        pos_rows = [(inc[i], w1[i] > 0) for i in range(inc.shape[0])]
        n_cl = inc.shape[0]
        used_total = int(inc.any(axis=1).sum())
    else:
        pos_rows, used_total, n_cl = [], 0, 0
        for c in range(len(tm.weight_banks)):
            inc = tmu_included(tm.clause_banks[c])
            w = np.asarray(tm.weight_banks[c].get_weights())
            used_total += int(inc.any(axis=1).sum())
            n_cl += inc.shape[0]
            # a clause votes for the positive class if it sits in the class-1 bank with w > 0 or the class-0 bank with w < 0
            pos_rows += [(inc[i], (w[i] > 0) if c == 1 else (w[i] < 0)) for i in range(inc.shape[0])]
    L = pos_rows[0][0].shape[0]
    pos_sets = [set(np.flatnonzero(r[: L // 2]).tolist()) for r, p in pos_rows if p and r.any()]
    return {"clauses": int(n_cl), "clauses_used": used_total, "group_word_coverage": group_word_coverage(pos_sets, world)}


# ------------------------------------------------------------------ sklearn (CPU)
def sk_fit_predict(kind: str, P: dict, seed: int, Xtr, ytr, Xev_list):
    import scipy.sparse as sp

    t0 = time.time()
    if kind == "lr":
        from tmlm.ssl import transfer as TR
        # public API of the transfer protocol: C by k-fold CV inside the training bags, then one refit; the
        # evaluation rows are the stacked evaluation sets, split back afterwards
        Xs = sp.csr_matrix(np.asarray(Xtr, np.float64))
        Xe = sp.vstack([sp.csr_matrix(np.asarray(X, np.float64)) for X in Xev_list]).tocsr()
        r = TR.lr_cell(Xs, np.asarray(ytr), Xe, np.zeros(Xe.shape[0], np.int64), seed, dense=False,
                       max_iter=int(P.get("max_iter", 1000)), c_grid=tuple(float(c) for c in P["c_grid"]),
                       max_folds=int(P.get("folds", 4)))
        cuts = np.cumsum([0] + [X.shape[0] for X in Xev_list])
        preds = [np.asarray(r.pred[cuts[i]:cuts[i + 1]]).astype(np.int64) for i in range(len(Xev_list))]
        return {"C": r.choice, "cv": r.cv}, preds, time.time() - t0, 0.0
    if kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=int(P["max_iter"]), learning_rate=float(P["learning_rate"]),
                                               max_leaf_nodes=int(P["max_leaf_nodes"]), early_stopping=False, random_state=int(seed))
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=tuple(P["hidden"]), alpha=float(P["alpha"]), max_iter=int(P["max_iter"]),
                              batch_size=int(P["batch_size"]), early_stopping=False, random_state=int(seed))
    else:
        raise ValueError(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.asarray(Xtr, np.float32), ytr)
    fit_s = time.time() - t0
    t1 = time.time()
    preds = [np.asarray(model.predict(np.asarray(X, np.float32))).astype(np.int64) for X in Xev_list]
    info = {"n_iter": int(getattr(model, "n_iter_", -1))}
    return info, preds, fit_s, time.time() - t1


# ------------------------------------------------------------------ one model on one seed (selection, refit, test)
PKG_KINDS = ("hier", "flat")
TMU_KINDS = ("tmu_tm", "tmu_ctm")
SK_KINDS = ("lr", "gbt", "mlp")


def family(kind: str) -> str:
    if kind in PKG_KINDS:
        return "pkg"
    if kind in TMU_KINDS:
        return "tmu"
    if kind in SK_KINDS:
        return "sk"
    raise ValueError(f"unknown model kind {kind!r}")


def run_model(model: str, spec: dict, seed: int, D: dict, htm=None) -> dict:
    kind = spec["kind"]
    fam = family(kind)
    grid = spec.get("grid", [{}])
    base = {k: v for k, v in spec.items() if k not in ("grid", "epochs", "kind")}
    epochs = int(spec.get("epochs", 1))
    tr, te, fi, va = D["train"], D["test"], D["fit"], D["val"]
    cell = {"seed": int(seed), "model": model, "family": fam, "kind": kind, "grid": grid, "epochs": epochs}

    def fit_pred(P, Xa, ya, Xs):
        if fam == "pkg":
            return pkg_fit_predict(htm, kind, P, seed, Xa, ya, Xs, epochs)
        if fam == "tmu":
            return tmu_fit_predict(kind, P, seed, Xa, ya, Xs, epochs)
        return sk_fit_predict(kind, P, seed, Xa, ya, Xs)

    choice = 0
    if len(grid) > 1:
        vals = []
        for gi, g in enumerate(grid):
            P = {**base, **g}
            obj, (pv,), fs, _ = fit_pred(P, fi.X, fi.y, [va.X])
            vals.append(float(np.mean(pv == va.y)))
            log(f"  seed {seed} {model} grid {gi} {g}: val {vals[-1]:.4f} ({fs:.0f} s)")
            del obj
        best = max(vals)
        choice = vals.index(best)
        cell["val_acc"] = vals
    P = {**base, **grid[choice]}
    cell["chosen"] = grid[choice]
    obj, (p_te, p_tr), fit_s, pred_s = fit_pred(P, tr.X, tr.y, [te.X, tr.X])
    cell.update({"test_acc": float(np.mean(p_te == te.y)), "train_acc": float(np.mean(p_tr == tr.y)), "fit_s": fit_s,
                 "predict_s": pred_s, "n_train": tr.n, "n_test": te.n,
                 "test_pred_sha256": hashing.sha256_bytes(p_te.astype("<i8").tobytes())})
    if fam == "pkg":
        cell.update(pkg_stats(obj, kind, P, D["world"], run_dir() / f"{model}_seed{seed}"))
        cell["state_sha256"] = pkg_state_sha(obj)
    elif fam == "tmu":
        cell.update(tmu_stats(obj, kind, D["world"]))
        cell["state_sha256"] = tmu_state_sha(obj, kind)
    else:
        cell["info"] = obj
    return cell


# ------------------------------------------------------------------ stages
def signatures(mods: list[str]) -> dict:
    import importlib

    out = {}
    for name in mods:
        mod_name, cls_name = name.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        out[name] = str(inspect.signature(cls.__init__))
    return out


def stage_models(cfg: dict, args, gpu: bool) -> int:
    P = cfg["params"]
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else P["seeds"])]
    models = {k: v for k, v in P["models"].items() if (family(v["kind"]) == "pkg") == gpu}
    if args.models:
        keep = set(args.models.split(","))
        models = {k: v for k, v in models.items() if k in keep}
    htm = None
    sig_names = []
    if gpu:
        htm = pkg_module()
        prov = pkg_provenance()
        M("package.provenance", prov)
        sig_names += ["PyHierarchicalTsetlinMachineCUDA.tm.TsetlinMachine"]
        M("package.grid_block", {"grid": list(GRID), "block": list(BLOCK)})
    else:
        sig_names += ["tmu.models.classification.vanilla_classifier.TMClassifier",
                      "tmu.models.classification.coalesced_classifier.TMCoalescedClassifier",
                      "sklearn.linear_model.LogisticRegression", "sklearn.ensemble.HistGradientBoostingClassifier",
                      "sklearn.neural_network.MLPClassifier"]
    sig = signatures(sig_names)
    (run_dir() / "signatures.json").write_text(json.dumps(sig, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    M("signatures_file", "signatures.json", sha256=hashing.sha256_file(run_dir() / "signatures.json"))
    flush()
    for seed in seeds:
        D = world_data(seed, P["world"])
        flush()
        if args.replay_check and seed == seeds[0]:
            replay_check(models, seed, D, htm, gpu)
        for model, spec in models.items():
            t0 = time.time()
            log(f"seed {seed} {model}: start")
            cell = run_model(model, spec, seed, D, htm)
            write_cell(cell)
            for k in ("test_acc", "train_acc", "fit_s", "clauses_used", "recovery_mean", "group_word_coverage", "chosen"):
                if k in cell:
                    M(f"seed{seed}.{model}.{k}", cell[k])
            if "printed_groups_equal_array_groups" in cell:
                check(f"seed{seed}.{model}.printed_groups_equal_array_groups", cell["printed_groups_equal_array_groups"])
            log(f"seed {seed} {model}: test {cell['test_acc']:.4f} train {cell['train_acc']:.4f} "
                f"({time.time() - t0:.0f} s) {({k: cell[k] for k in ('clauses_used', 'recovery_mean', 'group_word_coverage') if k in cell})}")
            flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


def replay_check(models: dict, seed: int, D: dict, htm, gpu: bool) -> None:
    """Protocol rule 4: the same model, seed and data twice (1 epoch on 1,000 bags) must give the same state hash."""
    sub = D["train"]
    Xs, ys = sub.X[:1000], sub.y[:1000]
    for model, spec in models.items():
        kind = spec["kind"]
        fam = family(kind)
        if fam == "sk":
            continue
        P = {**{k: v for k, v in spec.items() if k not in ("grid", "epochs", "kind")}, **spec.get("grid", [{}])[0]}
        hs = []
        for _ in range(2):
            if fam == "pkg":
                tm, _, _, _ = pkg_fit_predict(htm, kind, P, seed, Xs, ys, [], 1)
                hs.append(pkg_state_sha(tm))
            else:
                tm, _, _, _ = tmu_fit_predict(kind, P, seed, Xs, ys, [], 1)
                hs.append(tmu_state_sha(tm, kind))
            del tm
        check(f"replay.{model}.same_state_hash_twice", hs[0] == hs[1], sha=hs)


def stage_control(cfg: dict, args) -> int:
    C = cfg["params"]["control"]
    htm = pkg_module()
    M("package.provenance", pkg_provenance())
    seed = int(C["seed"])
    Xtr, ytr, ytr_clean = H.noisy_parity(int(C["n_train"]), seed, STREAM["parity_train"], float(C["noise"]))
    Xte, _, yte = H.noisy_parity(int(C["n_test"]), seed, STREAM["parity_test"], 0.0)
    M("control.train_noise_rate_measured", float(np.mean(ytr != ytr_clean)))
    M("control.data_sha256", {"Xtr": arr_sha(Xtr), "ytr": arr_sha(ytr), "Xte": arr_sha(Xte), "yte": arr_sha(yte)})
    hier = tuple((getattr(htm, t), int(n)) for t, n in C["hierarchy"])
    tm = htm.TsetlinMachine(int(C["clauses"]), int(C["T"]), float(C["s"]), number_of_state_bits=int(C["state_bits"]),
                            boost_true_positive_feedback=int(C["boost"]), hierarchy_structure=hier,
                            grid=GRID, block=BLOCK, seed=seed)
    curve, best, passed = [], 0.0, False
    t0 = time.time()
    for e in range(int(C["max_epochs"])):
        te0 = time.time()
        tm.fit(Xtr, ytr)
        ep_s = time.time() - te0
        if (e + 1) % int(C["eval_every"]) == 0 or e == 0:
            acc = float(np.mean(np.asarray(tm.predict(Xte)) == yte))
            best = max(best, acc)
            curve.append({"epoch": e + 1, "test_acc_clean": acc, "epoch_s": ep_s, "elapsed_s": time.time() - t0})
            log(f"control epoch {e + 1}: clean test {acc:.4f} (epoch {ep_s:.1f} s)")
            M("control.curve", curve)
            flush()
            if acc >= float(C["pass_threshold"]):
                passed = True
                break
        if time.time() - t0 > float(C["wall_cap_s"]):
            log("control: wall cap reached")
            break
    M("control.best_test_acc_clean", best)
    M("control.passed", passed, note=f"pass = clean test accuracy >= {C['pass_threshold']} within {C['max_epochs']} epochs and {C['wall_cap_s']} s")
    M("control.epochs_run", curve[-1]["epoch"] if curve else 0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tm.print_hierarchy()
    (run_dir() / "control_hierarchy.txt").write_text(buf.getvalue(), encoding="utf-8")
    inc = pkg_included(tm)
    check("control.render_equals_print_hierarchy", H.render_hierarchy(inc, [tuple(x) for x in hier], True) == buf.getvalue())
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ report
def t_int(v) -> dict:
    from tmlm.ssl import transfer as TR
    return TR.t_interval(list(v))


def wilcoxon_greater(d) -> float | None:
    d = np.asarray(d, np.float64)
    if d.size < 2 or np.all(d == 0):
        return None
    from scipy.stats import wilcoxon
    return float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)


def verdict_h17_5(cells: list[dict], seeds: list[int], V: dict) -> dict:
    """H17.5 per registered pair: hierarchical mean >= hier_min AND mean(hier - flat) >= margin over ALL seeds
    (a pair with a missing seed is 'incomplete', never a pass)."""
    by = {(c["seed"], c["model"]): c for c in cells}
    out = {}
    for role, (hier, flat) in V["pairs"].items():
        common = [s for s in seeds if (s, hier) in by and (s, flat) in by]
        if len(common) < len(seeds):
            out[role] = {"status": "incomplete", "seeds_present": common}
            continue
        h = np.asarray([by[(s, hier)]["test_acc"] for s in common])
        f = np.asarray([by[(s, flat)]["test_acc"] for s in common])
        d = h - f
        out[role] = {"hier": hier, "flat": flat, "hier_mean": float(h.mean()), "flat_mean": float(f.mean()),
                     "diff": t_int(d), "wilcoxon_one_sided_p": wilcoxon_greater(d),
                     "meets_H17_5": bool(h.mean() >= float(V["hier_min"]) and d.mean() >= float(V["margin"]))}
    return out


def stage_report(cfg: dict, args) -> int:
    P = cfg["params"]
    cells, control = [], {}
    for d in args.runs.split(","):
        f = Path(d) / "cells.jsonl"
        if f.exists():
            cells += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        mf = Path(d) / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            if "control.passed" in m:
                control = {k: m[k]["value"] for k in ("control.passed", "control.best_test_acc_clean", "control.epochs_run") if k in m}
    seeds = [int(s) for s in P["seeds"]]
    by = {(c["seed"], c["model"]): c for c in cells}
    table = {}
    for model in sorted({c["model"] for c in cells}):
        acc = [by[(s, model)]["test_acc"] for s in seeds if (s, model) in by]
        table[model] = {"n_seeds": len(acc), "test_acc": t_int(acc), "min": min(acc) if acc else None, "max": max(acc) if acc else None}
        for k in ("recovery_mean", "clauses_used", "group_word_coverage"):
            vals = [by[(s, model)][k] for s in seeds if (s, model) in by and k in by[(s, model)]]
            if vals:
                table[model][k] = t_int(vals)
    M("table", table, label="derived")
    verdicts = verdict_h17_5(cells, seeds, P["verdict"])
    M("verdict.H17_5", verdicts, label="derived", rule=P["verdict"]["rule"])
    M("positive_control", control if control else "no control run given", label="derived")
    flush()
    log(json.dumps({"verdicts": verdicts, "control": control}, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gpu", "cpu", "control", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None, help="override the config's seed list (comma separated)")
    ap.add_argument("--models", default=None, help="subset of the config's models (comma separated)")
    ap.add_argument("--runs", default=None, help="report: comma separated run dirs holding cells.jsonl")
    ap.add_argument("--replay-check", action="store_true", help="train each Tsetlin model twice (1 epoch, 1,000 bags) and compare state hashes")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    M("config_file", args.config, sha256=hashing.sha256_file(args.config))
    M("stage", args.stage)
    try:
        if args.stage == "gpu":
            return stage_models(cfg, args, gpu=True)
        if args.stage == "cpu":
            return stage_models(cfg, args, gpu=False)
        if args.stage == "control":
            return stage_control(cfg, args)
        return stage_report(cfg, args)
    finally:
        flush()


if __name__ == "__main__":
    sys.exit(main())
