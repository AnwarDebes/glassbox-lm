"""EXPLORATORY hierarchy diagnostics on the H-AO world (docs/13 amendment A8, item 3, rows (i) to (vi); A7 H17.5,
phase R2). World seed 100 first, then 101 (docs/13 section 3: seeds >= 100 are exploratory and never reported as
results). Config experiments/cvtm/configs/ssl_r2_hao_diagnostics.json; its reading notes were written before any
output of this script existed.

Why: on H-AO no tuned configuration of the hierarchical package reached 0.95 training accuracy at either pinned
commit (experiments/cvtm/configs/ssl_r2_hao_tuned.json, c85ca4e7; ssl_r2_hao_head_tuned.json, HEAD 5b4a223), and the
flat control reached 0.80 or more on validation. Before any H17.5 seed is read, A8 asks WHY: does the package pass
its author's own demos, does what it learns transfer to unseen combinations, is the true hierarchy even a stable
state of its learning rule, how many epochs would it need, and can a non-TA logic learner (Valiant elimination)
learn the same concept from the same bags.

Data (per world seed; hao_world with the streams of experiments/ssl_r2_hao.py, via ssl_r2_hao_tune.tuning_data):
  fit      8,000 bags from the FIT combinations (the training combinations minus a carved 20 percent), stream 3; the
           only bags any learner here is trained on. Seed 100: fit and validation bags are checked bit-identical to
           results/20260924T215754Z_ssl_r2_hao_gpu_exploratory (the data of every tuning run).
  seen     4,000 FRESH bags from the fit combinations (stream 21): held-out bags of SEEN combinations.
  unseen   the 4,000 validation bags (stream 4) of the carved validation combinations: every combination contained
           in such a positive is a validation combination (hao_world's construction), so no fit bag contains it:
           held-out bags of UNSEEN combinations. At world seed 100 these are EXACTLY the validation bags on which the
           tuned configurations were chosen, so seed-100 unseen numbers are biased upward by that choice; seed 101
           is the unbiased reading (config, reading_notes.seen_unseen).
  unseen_fresh  4,000 FRESH bags from the same validation combinations (stream 22), never used by any tuning run:
           reported next to unseen at both seeds (at seed 100 the combinations were still part of the choice, the
           bags were not).
  The registered test combinations (70 percent of all combinations) are never sampled and never scored.

Stages (--stage). Package stages run under the venv of one package commit (.venv-tm: c85ca4e7, .venv-htm: HEAD
5b4a223); the script reads the installed commit, picks that commit's tuned config and demo defaults, and refuses a
mismatch. Package constructors are those of the tuning scripts (ssl_r2_hao.pkg_make on c85ca4e7,
ssl_r2_hao_head.pkg_make_head with HEAD's four switches on 5b4a223; grid (208,1,1), block (128,1,1), machine seed =
world seed; protocol rule 4 is checked with --replay-check).
  demos        (vi)  Granmo's examples/ANDOfXORsDemo.py and examples/MultiConceptDemo.py at the DEFAULTS OF THE
                     INSTALLED COMMIT (the argparse defaults are parsed from that commit's source in
                     third_party/ref/PyHierarchicalTsetlinMachineCUDA and must equal the config; source sha256
                     checked), with his run counts, test accuracy after every epoch (evaluation draws no random
                     numbers, so it does not change training). Data generators ported with seeded RandomStates
                     (ssl_r2_hao_head.andxor_data; hao_diag.multiconcept_data); machine seeds drawn as the demos draw
                     them, from a seeded RandomState; MultiClassTsetlinMachine.fit draws from the GLOBAL numpy
                     generator, which is seeded with the machine seed before each run.
  seen_unseen  (ii)  the tuned negated-literal pair (htm_hier_neg, htm_flat_neg) of the installed commit's tuned
                     config at its tuned epochs, trained on fit; every 5 epochs: accuracy on fit, seen, unseen and
                     unseen_fresh (overall, positives, negatives), leaf statistics and OR-group read-outs of the tuning
                     scripts. At world seed 100 and the tuned epoch the fit and unseen accuracies must equal the tuning
                     run's train_acc_at_chosen_epoch and val_acc_at_chosen_epoch (check ...reproduces_tuning_run).
  pospair      (i)   the same for the positive-literal pair (htm_hier_pos, htm_flat_pos: append_negated False), plus
                     derived capacity facts (a flat monotone clause is one conjunction, the concept's minimal monotone
                     DNF has G^K = 1,296 terms) and the pair differences.
  oracle       (iii) the TRUE hierarchy (hao_diag.oracle_hierarchy; de_morgan negative clauses with negated
                     literals, a bias clause with positive literals only) written into a fresh machine of each tuned
                     hierarchical configuration by DIRECT device copies into ta_state_hierarchy_gpu and
                     clause_weights_gpu after the package's own allocate_gpu_memory and
                     initialize_weights_and_ta_states, with `first` cleared so fit() keeps the state: exactly the
                     first fit()'s allocation and initialisation, then the copies. The package's load(state_dict)
                     (both commits: allocate_gpu_memory, the same two memcpy_htod, first = False) would give the same
                     state; it skips initialize_weights_and_ta_states, whose prepare_weights draws no random number for
                     a non-coalesced TsetlinMachine and whose prepare_hierarchy output the copy overwrites, so the two
                     routes differ in no buffer that the oracle does not overwrite; the direct copies are kept because
                     they reproduce the first fit() step by step and are verified by read-back. The package's
                     set_state is not used: at both commits it allocates clauses x teams x state bits x 4 bytes, a
                     factor literal-chunks-per-leaf too small, so its memcpy_htod of a full-size state writes past the
                     end of that allocation (an out-of-bounds device write: undefined behaviour, a CUDA error or silent
                     corruption of neighbouring buffers), and it does not clear `first`, so the next fit() re-allocates
                     and re-initialises (audited from the installed source at run time and in tests on both sources).
                     Verified by read-back (raw words, weights,
                     top-plane includes, the package's own get_ta_states kernel), by the package's class sums equal
                     to the numpy reference on every bag, and by predicting fit, seen and unseen bags (at least 0.95
                     required, 1.00 expected). Then trained on fit WITHOUT re-clamping at s in {4, 10, 25} for 50
                     epochs (other hyperparameters as tuned), state depths "deep" and "boundary"; every 5 epochs:
                     accuracy, per-node Jaccard with the oracle's node identities, literals added and removed.
  threshold    (iv)  one instance (world seed 100): the best tuned hierarchical configuration (highest validation
                     accuracy at its chosen epoch over both tuned configs; HEAD htm_hier_neg, 0.750 vs 0.708) trained
                     until 0.95 fit accuracy or 2 GPU-hours (training plus evaluation wall time), every 10 epochs.
  valiant      (v)   CPU. Valiant elimination (hao_diag.valiant_fit) on the POSITIVE fit bags: word frequency over the
                     positives, keep frequency >= 0.1, every OR of up to 6 kept words, delete each OR that some
                     positive bag misses, predict positive iff every surviving OR is hit. Accuracy on fit, seen and
                     unseen; the minimal surviving ORs against the true groups (exact matches, Jaccard under the best
                     matching). Also over growing numbers of positives (a sample-size curve).
  controls     (ii)  CPU. The non-package rows of the c85ca4e7 tuned config (tmu_tm, tmu_ctm, lr, gbt, mlp: the
                     constructors of experiments/ssl_r2_hao.py) trained on fit only and scored on fit, seen, unseen and
                     unseen_fresh; a tmu grid is fixed to the point the registered H17.5 CPU stage chose at seed 100
                     (params.controls.grid_choice); at seed 100 the tmu unseen accuracy must equal that run's logged
                     validation accuracy (check ...reproduces_selection_run).
  selftest           GPU, tiny: --expect-tag c85|head is checked against the detected package commit; then the oracle
                     write round trip on tiny instances (full and partly filled last literal chunk, depths deep and
                     boundary, both literal sets), the oracle, seen_unseen, pospair and threshold stages on a tiny world
                     with a tiny tuned config of that commit (their cells go to <run dir>/selftest/, never into the
                     run's own cells.jsonl), two seeded runs of each demo variant on tiny defaults (equal curves) and
                     the demo source checks of the real config. The chain runs it before the HEAD stages.
  report             CPU. Reads the cells.jsonl of the given run dirs and applies the config's reading notes.

Launch: the chain chain_hao_diag.sh (CPU slot first, the shared GPU lock inside each step, docs/12 F25); one step:
    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/mem_guard.py --scope group \\
        --limit-gib 6 -- scripts/run_logged.py --name ssl_r2_hao_diag_oracle_c85_seed100_exploratory \\
        --config experiments/cvtm/configs/ssl_r2_hao_diagnostics.json -- .venv-tm/bin/python -u \\
        experiments/ssl_r2_hao_diagnostics.py --config experiments/cvtm/configs/ssl_r2_hao_diagnostics.json \\
        --stage oracle --world-seed 100 --replay-check
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import hao_diag as HG  # noqa: E402
from tmlm.ssl import hao_world as H  # noqa: E402

import ssl_r2_hao as E  # noqa: E402  (grid/block, include decoding, weights, state hash, provenance, c85 constructor)
import ssl_r2_hao_head as HD  # noqa: E402  (HEAD constructor with switches, OR read-outs, ANDOfXORs generator)
import ssl_r2_hao_tune as TU  # noqa: E402  (fit/validation data of the tuning runs, leaf statistics)

C85 = "c85ca4e70af90511d6b4f8e8eedccd23a7d7d0ec"
COMMITS = {C85: "c85", HD.PIN: "head"}
REF_REPO = ROOT / "third_party" / "ref" / "PyHierarchicalTsetlinMachineCUDA"
SETS = ("fit", "seen", "unseen", "unseen_fresh")
T0 = time.time()
METRICS: dict = {}
FAILED: list = []
_OUT: list = []


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def jsonable(x):
    """Plain JSON (numpy scalars and arrays converted, int dict keys turned into strings)."""
    def fix(v):
        if isinstance(v, dict):
            return {str(k): fix(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(w) for w in v]
        return v
    return fix(json.loads(json.dumps(x, default=_np_default)))


def M(name: str, value, label: str = "measured", **extra) -> None:
    METRICS[name] = {"value": jsonable(value), "label": label, **jsonable(extra)}


def check(name: str, ok: bool, **extra) -> bool:
    M(f"check.{name}", bool(ok), **extra)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {json.dumps(jsonable(extra))[:400]}")
    return bool(ok)


def flush() -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_metrics(METRICS, on_conflict="replace")


def out_dir() -> Path:
    if not _OUT:
        d = runlog.current_run_dir()
        if d is None:
            d = Path(os.environ.get("HAO_DIAG_OUT") or tempfile.mkdtemp(prefix="ssl_r2_hao_diag_"))
        Path(d).mkdir(parents=True, exist_ok=True)
        _OUT.append(Path(d))
    return _OUT[0]


def append_row(row: dict) -> None:
    with open(out_dir() / "cells.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


# ------------------------------------------------------------------ package and tuned configurations
def load_package():
    htm = E.pkg_module()  # creates the CUDA context
    prov = E.pkg_provenance()
    commit = ((prov.get("direct_url") or {}).get("vcs_info") or {}).get("commit_id")
    tag = COMMITS.get(commit)
    if tag is None:
        raise SystemExit(f"PyHierarchicalTsetlinMachineCUDA at {commit}: expected one of {sorted(COMMITS)}")
    sig = str(inspect.signature(htm.TsetlinMachine.__init__))
    if tag == "head" and not all(s in sig for s in HD.SWITCHES):
        raise SystemExit(f"HEAD package without the switches {HD.SWITCHES}: {sig}")
    M("package.provenance", prov)
    M("package.tag", tag)
    M("package.TsetlinMachine_signature", sig)
    M("package.grid_block", {"grid": list(E.GRID), "block": list(E.BLOCK)})
    return htm, tag


def make_model(htm, tag: str, kind: str, n_features: int, P: dict, seed: int):
    if tag == "head":
        return HD.pkg_make_head(htm, kind, n_features, P, seed)
    on = [k for k in HD.SWITCHES if P.get(k)]
    if on:
        raise ValueError(f"switches {on} exist only at HEAD 5b4a223")
    return E.pkg_make(htm, kind, n_features, P, seed)


def tuned_config(cfg: dict, tag: str) -> tuple[str, dict]:
    rel = cfg["params"]["tuned_configs"][tag]
    T = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    if tag == "head" and (T.get("package") or {}).get("commit") != HD.PIN:
        raise SystemExit(f"{rel} is not the HEAD {HD.PIN} tuned config")
    M(f"tuned_config.{tag}", rel, sha256=hashing.sha256_file(ROOT / rel))
    return rel, T


def model_params(spec: dict) -> dict:
    grid = spec.get("grid", [{}])
    if len(grid) != 1:
        raise ValueError("a tuned model has exactly one grid point")
    return {**{k: v for k, v in spec.items() if k not in ("grid", "epochs", "kind")}, **grid[0]}


def best_hier_config(cfg: dict) -> dict:
    """Highest validation accuracy at the chosen epoch among the hierarchical models of both tuned configs (ties:
    the order of params.tuned_configs)."""
    cands = []
    for order, (tag, rel) in enumerate(cfg["params"]["tuned_configs"].items()):
        T = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        for m, ch in T["tuning"]["chosen"].items():
            if T["params"]["models"][m]["kind"] == "hier":
                cands.append({"tag": tag, "model": m, "val_acc": float(ch["val_acc_at_chosen_epoch"]),
                              "train_acc": float(ch["train_acc_at_chosen_epoch"]), "tuning_config": ch["tuning_config"],
                              "order": order})
    best = max(cands, key=lambda c: (c["val_acc"], -c["order"]))
    return {"best": best, "candidates": cands}


# ------------------------------------------------------------------ data
def group_word_max(world: H.HaoWorld, bags: H.HaoBags) -> int:
    rows = np.repeat(np.arange(bags.n), np.diff(bags.indptr))
    g = world.word_group[bags.indices]
    keep = g >= 0
    if not keep.any():
        return 0
    return int(np.bincount(rows[keep] * world.n_groups + g[keep]).max())


def world_sets(cfg: dict, seed: int) -> dict:
    P = cfg["params"]
    W = P["world"]
    if seed < 100:
        raise SystemExit("exploratory only: world seeds >= 100 (docs/13 section 3)")
    d = TU.tuning_data(seed, W)
    world, fit_mask, val_mask = d["world"], d["fit_mask"], d["val_mask"]
    kw = {"n_distractors": int(W["n_distractors"]), "extra_probs": tuple(W["extra_probs"])}
    seen = H.sample_bags(world, int(P["n_seen"]), fit_mask, int(P["streams"]["seen"]), **kw)
    fresh = H.sample_bags(world, int(P["n_unseen_fresh"]), val_mask, int(P["streams"]["unseen_fresh"]), **kw)
    S = {"world": world, "fit": d["fit"], "seen": seen, "unseen": d["val"], "unseen_fresh": fresh, "fit_mask": fit_mask,
         "val_mask": val_mask}
    for name, allowed in (("fit", fit_mask), ("seen", fit_mask), ("unseen", val_mask), ("unseen_fresh", val_mask)):
        ok = H.check_bags(world, S[name], allowed, int(W["n_distractors"]))
        check(f"seed{seed}.data.{name}.construction", all(ok.values()), detail=ok)
        M(f"seed{seed}.data.{name}.n", S[name].n)
        M(f"seed{seed}.data.{name}.X_sha256", E.arr_sha(S[name].X))
        M(f"seed{seed}.data.{name}.y_sha256", E.arr_sha(S[name].y))
    check(f"seed{seed}.data.no_test_combination_used", not bool(np.any((fit_mask | val_mask) & ~world.train_combo)))
    check(f"seed{seed}.data.fit_and_unseen_combinations_disjoint", not bool(np.any(fit_mask & val_mask)))
    ref = P["reference"]
    if seed == int(ref["seed"]):
        for name, key in (("fit", "fit"), ("unseen", "val")):
            check(f"seed{seed}.data.{name}.equals_reference_run",
                  E.arr_sha(S[name].X) == ref[key]["X_sha256"] and E.arr_sha(S[name].y) == ref[key]["y_sha256"],
                  reference_run=ref["run"])
    M(f"seed{seed}.world.groups", world.groups.tolist())
    M(f"seed{seed}.world.fit_combos", int(fit_mask.sum()))
    M(f"seed{seed}.world.unseen_combos", int(val_mask.sum()))
    M(f"seed{seed}.data.max_words_of_one_group_in_a_bag", max(group_word_max(world, S[n]) for n in SETS),
      note="the bias/never leaves of the positive-literal oracle hold all G words of a group; valid iff this is < G")
    M(f"seed{seed}.data.test_combinations_used", False, note="the registered test combinations are never sampled or scored")
    M(f"seed{seed}.data.unseen_is_tuning_validation", seed == int(ref["seed"]),
      note="at the reference seed the unseen bags are the validation bags the tuned configurations were chosen on "
           "(biased upward); unseen_fresh (stream 22) are new bags of the same combinations")
    return S


def scores(tm, X: np.ndarray) -> np.ndarray:
    return np.asarray(tm.score(np.ascontiguousarray(X, dtype=np.uint32)), dtype=np.float64)


def pred_metrics(pred: dict, S: dict) -> dict:
    """Accuracy (overall, positives, negatives), positive rate and the gaps, from bool predictions per set name."""
    out = {}
    names = list(pred)
    for name in names:
        p = np.asarray(pred[name]).astype(bool)
        y = S[name].y.astype(bool)
        out[f"{name}_acc"] = float(np.mean(p == y))
        out[f"{name}_pos_acc"] = float(np.mean(p[y])) if y.any() else None
        out[f"{name}_neg_acc"] = float(np.mean(~p[~y])) if (~y).any() else None
        out[f"{name}_pos_rate"] = float(p.mean())
    if "seen" in names and "unseen" in names:
        out["gap_seen_minus_unseen"] = out["seen_acc"] - out["unseen_acc"]
    if "seen" in names and "unseen_fresh" in names:
        out["gap_seen_minus_unseen_fresh"] = out["seen_acc"] - out["unseen_fresh_acc"]
    if "fit" in names and "seen" in names:
        out["gap_fit_minus_seen"] = out["fit_acc"] - out["seen_acc"]
    return out


def eval_sets(tm, S: dict, names=SETS) -> dict:
    return pred_metrics({name: scores(tm, S[name].X) >= 0 for name in names}, S)  # positive at class sum >= 0


# ------------------------------------------------------------------ training curve (all package stages)
def row_line(name: str, r: dict) -> str:
    s = (f"  {name} ep {r['epoch']}: fit {r['fit_acc']:.4f} seen {r['seen_acc']:.4f} unseen {r['unseen_acc']:.4f} "
         f"fresh {r['unseen_fresh_acc']:.4f} "
         f"(unseen+ {r['unseen_pos_acc']:.3f} unseen- {r['unseen_neg_acc']:.3f}) Rpos {r.get('R_pos_pool', -1):.2f} "
         f"Rdm {r.get('R_dm_pool', -1):.2f} used {r.get('pos_clauses_used', -1)}+/{r.get('neg_clauses_used', -1)}-")
    if "pos_node_jaccard_mean" in r:
        s += f" Jnode+ {r['pos_node_jaccard_mean']:.3f} exact {r['clauses_exact_share']:.3f}"
    if "dm_node_jaccard_mean" in r:
        s += f" Jdm {r['dm_node_jaccard_mean']:.3f}"
    return s + f" ({r['fit_s_cum']:.0f} s)"


def train_curve(htm, tag: str, name: str, kind: str, P: dict, S: dict, seed: int, epochs: int, every: int,
                cap_s: float, stop_at: float | None = None, oracle: HG.OracleHierarchy | None = None, tm=None,
                save_hierarchy: bool = True) -> tuple[dict, object]:
    world = S["world"]
    n_features = int(world.vocab)
    if tm is None:
        tm = make_model(htm, tag, kind, n_features, P, seed)
    Xf = np.ascontiguousarray(S["fit"].X, dtype=np.uint32)
    yf = np.ascontiguousarray(S["fit"].y, dtype=np.uint32)
    cell = {"name": name, "kind": kind, "params": P, "seed": int(seed), "epochs_planned": int(epochs),
            "eval_every": int(every), "curve": [], "stop_reason": "epochs",
            "machine": {"seed": int(seed), "grid": list(E.GRID), "block": list(E.BLOCK)}}

    def row(epoch: int, fit_s: float, wall: float) -> dict:
        r = {"epoch": int(epoch), "fit_s_cum": fit_s, "wall_s": wall, **eval_sets(tm, S),
             **TU.leaf_stats(tm, kind, P, world), **HD.or_readouts(tm, kind, P, world)}
        if oracle is not None:
            r.update(HG.node_recovery(E.pkg_included(tm), oracle))
        return r

    t0 = time.time()
    fit_s = 0.0
    if oracle is not None:
        cell["curve"].append(row(0, 0.0, 0.0))
        log(row_line(name, cell["curve"][-1]))
    cell["epochs_run"] = 0
    for e in range(int(epochs)):
        tf = time.time()
        tm.fit(Xf, yf)
        fit_s += time.time() - tf
        cell["epochs_run"] = e + 1
        last = e + 1 == int(epochs)
        if time.time() - t0 > float(cap_s):
            last, cell["stop_reason"] = True, "wall_cap"
        if (e + 1) % int(every) == 0 or (e == 0 and oracle is None) or last:
            r = row(e + 1, fit_s, time.time() - t0)
            cell["curve"].append(r)
            log(row_line(name, r))
            if stop_at is not None and r["fit_acc"] >= float(stop_at) and not last:
                last, cell["stop_reason"] = True, f"fit_acc >= {stop_at}"
            if time.time() - t0 > float(cap_s) and not last:
                last, cell["stop_reason"] = True, "wall_cap"
        if last:
            break
    final = cell["curve"][-1]
    cell.update({"final": {k: v for k, v in final.items() if not isinstance(v, (list, dict))}, "fit_s": fit_s,
                 "wall_s": time.time() - t0, "state_sha256": E.pkg_state_sha(tm)})
    if kind == "hier" and save_hierarchy:
        inc = E.pkg_included(tm)
        hier = [tuple(x) for x in E.pkg_hierarchy(htm, kind, n_features, P)]
        fn = out_dir() / f"{name.replace('/', '_')}_seed{seed}.hierarchy.txt.gz"
        with gzip.open(fn, "wt", encoding="utf-8") as fh:
            fh.write(H.render_hierarchy(inc, hier, bool(P["append_negated"])))
        cell["hierarchy_file"] = fn.name
    return cell, tm


def replay_check(htm, tag: str, name: str, kind: str, P: dict, S: dict, seed: int, prepare=None, n: int = 1000) -> None:
    """Protocol rule 4: the same machine, seed, grid, block and data twice (1 epoch on n fit bags): the same state hash."""
    X = np.ascontiguousarray(S["fit"].X[:n], dtype=np.uint32)
    y = np.ascontiguousarray(S["fit"].y[:n], dtype=np.uint32)
    hs = []
    for _ in range(2):
        tm = make_model(htm, tag, kind, int(S["world"].vocab), P, seed)
        if prepare is not None:
            prepare(tm)
        tm.fit(X, y)
        hs.append(E.pkg_state_sha(tm))
        del tm
    check(f"replay.{name}.same_state_hash_twice", hs[0] == hs[1], sha=hs)


# ------------------------------------------------------------------ stages (ii) and (i): seen vs unseen combinations
def summary_keys(cell: dict) -> dict:
    f = cell["final"]
    keys = ("fit_acc", "seen_acc", "unseen_acc", "seen_pos_acc", "unseen_pos_acc", "unseen_neg_acc",
            "unseen_fresh_acc", "unseen_fresh_pos_acc", "unseen_fresh_neg_acc",
            "gap_seen_minus_unseen", "gap_seen_minus_unseen_fresh", "gap_fit_minus_seen", "R_pos_pool", "R_dm_pool",
            "R_node_max")
    return {k: f[k] for k in keys if k in f}


def tuning_reproduction(row: dict, choice: dict, n_fit: int, n_unseen: int) -> tuple[bool, dict]:
    """Whether an evaluated row reproduces the tuning run at its chosen epoch: fit accuracy equal to
    train_acc_at_chosen_epoch and unseen accuracy equal to val_acc_at_chosen_epoch, compared as correct-bag counts
    (both sides are count / n)."""
    got = {"fit": int(round(float(row["fit_acc"]) * n_fit)), "unseen": int(round(float(row["unseen_acc"]) * n_unseen))}
    want = {"fit": int(round(float(choice["train_acc_at_chosen_epoch"]) * n_fit)),
            "unseen": int(round(float(choice["val_acc_at_chosen_epoch"]) * n_unseen))}
    detail = {"epoch": int(row["epoch"]), "fit_acc": row["fit_acc"], "unseen_acc": row["unseen_acc"],
              "tuning_train_acc": choice["train_acc_at_chosen_epoch"], "tuning_val_acc": choice["val_acc_at_chosen_epoch"],
              "chosen_epoch": choice.get("chosen_epoch"), "tuning_config": choice.get("tuning_config"),
              "tuning_run": choice.get("tuning_run"), "correct_bags": got, "correct_bags_tuning": want}
    return got == want, detail


def check_tuning_reproduction(cfg: dict, pre: str, seed: int, cell: dict, choice: dict | None, S: dict) -> None:
    """At the reference seed (the tuning data, checked bit-identical by world_sets) and the tuned chosen epoch, the
    run must reproduce the tuning run (check <pre>.reproduces_tuning_run); otherwise the reason it is not checked."""
    reason = None
    if seed != int(cfg["params"]["reference"]["seed"]):
        reason = f"world seed {seed} is not the tuning seed {cfg['params']['reference']['seed']}"
    elif not choice or "chosen_epoch" not in choice:
        reason = "no tuning choice with a chosen_epoch for this model"
    row = next((r for r in cell["curve"] if int(r["epoch"]) == int(choice["chosen_epoch"])), None) if reason is None else None
    if reason is None and row is None:
        reason = f"no evaluated epoch {choice['chosen_epoch']} (epochs run {cell['epochs_run']}, stop {cell['stop_reason']})"
    if reason is not None:
        M(f"{pre}.reproduces_tuning_run", None, label="derived", reason=reason)
        return
    ok, detail = tuning_reproduction(row, choice, S["fit"].n, S["unseen"].n)
    check(f"{pre}.reproduces_tuning_run", ok, **detail)


def stage_models(cfg: dict, args, stage: str) -> int:
    htm, tag = load_package()
    seed = int(args.world_seed)
    S = world_sets(cfg, seed)
    rel, T = tuned_config(cfg, tag)
    Q = cfg["params"][stage]
    models = args.models.split(",") if args.models else list(Q["models"])
    flush()
    cells = {}
    for i, model in enumerate(models):
        spec = T["params"]["models"][model]
        kind = spec["kind"]
        P = model_params(spec)
        if args.replay_check and i == 0:
            replay_check(htm, tag, f"{stage}.{model}", kind, P, S, seed)
        log(f"{stage} {tag} seed {seed} {model}: {json.dumps(P)} epochs {spec['epochs']}")
        cell, tm = train_curve(htm, tag, model, kind, P, S, seed, int(spec["epochs"]), int(cfg["params"]["eval_every"]),
                               float(Q["wall_cap_s"]))
        del tm
        choice = T["tuning"]["chosen"].get(model)
        if cell["epochs_run"] == int((choice or {}).get("chosen_epoch", -1)):
            check_tuning_reproduction(cfg, f"{stage}.{tag}.seed{seed}.{model}", seed, cell, choice, S)
        else:
            M(f"{stage}.{tag}.seed{seed}.{model}.reproduces_tuning_run", None, label="derived",
              reason=f"epochs run {cell['epochs_run']} != tuned chosen epoch {(choice or {}).get('chosen_epoch')}")
        cell.update({"stage": stage, "tag": tag, "world_seed": seed, "model": model, "tuned_config": rel,
                     "tuned_epochs": int(spec["epochs"]), "tuning_choice": choice})
        append_row(cell)
        cells[model] = cell
        for k, v in summary_keys(cell).items():
            M(f"{stage}.{tag}.seed{seed}.{model}.final.{k}", v)
        M(f"{stage}.{tag}.seed{seed}.{model}.epochs_run", cell["epochs_run"])
        flush()
    if stage == "pospair":
        pospair_derived(cfg, T, S, cells, tag, seed)
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


def pospair_derived(cfg: dict, T: dict, S: dict, cells: dict, tag: str, seed: int) -> None:
    world = S["world"]
    G, K = world.group_size, world.n_groups
    fl = T["params"]["models"].get("htm_flat_pos")
    hi = T["params"]["models"].get("htm_hier_pos")
    cap = {"minimal_monotone_dnf_terms": int(G ** K), "fit_combinations": int(S["fit_mask"].sum()),
           "unseen_combinations": int(S["val_mask"].sum())}
    if fl is not None:
        n_flat = int(fl["clauses"]) * int(fl.get("alternatives", 1)) * int(fl.get("n_concepts", 1))
        cap["flat_clauses"] = n_flat
        cap["flat_positive_clauses"] = n_flat // 2
        cap["flat_can_hold_the_full_dnf"] = n_flat // 2 >= G ** K
        cap["flat_can_hold_every_fit_combination"] = n_flat // 2 >= int(S["fit_mask"].sum())
    if hi is not None:
        cap["hier_alternatives"] = int(hi["alternatives"])
        cap["one_hier_clause_can_hold_the_concept"] = int(hi["alternatives"]) >= G and int(hi["n_concepts"]) == K
    M(f"pospair.{tag}.seed{seed}.capacity", cap, label="derived",
      note="a flat positive-literal clause is one conjunction; the minimal monotone DNF of AND over 4 groups of an OR "
           "of 6 words has 6^4 terms; a flat machine that holds only 4-word conjunctions of SEEN combinations fires "
           "on no unseen positive (every combination in an unseen positive is a validation combination)")
    if "htm_hier_pos" in cells and "htm_flat_pos" in cells:
        h, f = cells["htm_hier_pos"]["final"], cells["htm_flat_pos"]["final"]
        M(f"pospair.{tag}.seed{seed}.hier_minus_flat", {k: h[k] - f[k] for k in ("fit_acc", "seen_acc", "unseen_acc")},
          label="derived")


# ------------------------------------------------------------------ stage (ii) controls: the non-package tuned rows
def stage_controls(cfg: dict, args) -> int:
    """tmu_tm, tmu_ctm, lr, gbt, mlp of the tuned config (constructors of ssl_r2_hao.py), trained on fit only."""
    seed = int(args.world_seed)
    S = world_sets(cfg, seed)
    Q = cfg["params"]["controls"]
    rel, T = tuned_config(cfg, Q["tuned_config"])
    flush()
    for model in (args.models.split(",") if args.models else list(Q["models"])):
        spec = T["params"]["models"][model]
        kind, grid = spec["kind"], spec.get("grid", [{}])
        if len(grid) > 1 and model not in Q["grid_choice"]:
            raise SystemExit(f"{model}: a grid of {len(grid)} points needs params.controls.grid_choice")
        P = {**{k: v for k, v in spec.items() if k not in ("grid", "epochs", "kind")}, **Q["grid_choice"].get(model, grid[0])}
        Xs = [S[n].X for n in SETS]
        if kind in E.TMU_KINDS:
            _, preds, fit_s, _ = E.tmu_fit_predict(kind, P, seed, S["fit"].X, S["fit"].y, Xs, int(spec["epochs"]))
        else:
            _, preds, fit_s, _ = E.sk_fit_predict(kind, P, seed, S["fit"].X, S["fit"].y, Xs)
        f = pred_metrics(dict(zip(SETS, preds)), S)
        cell = {"stage": "controls", "tag": "cpu", "world_seed": seed, "model": model, "kind": kind, "params": P,
                "epochs": spec.get("epochs"), "final": f, "fit_s": fit_s, "tuned_config": rel}
        want = Q["grid_choice_val_acc_seed100"].get(model)
        if want is not None and seed == int(cfg["params"]["reference"]["seed"]):
            n = S["unseen"].n
            check(f"controls.seed{seed}.{model}.reproduces_selection_run", round(f["unseen_acc"] * n) == round(float(want) * n),
                  unseen_acc=f["unseen_acc"], logged_val_acc=want, run=Q["grid_choice_source"])
        append_row(cell)
        for k, v in summary_keys(cell).items():
            M(f"controls.seed{seed}.{model}.final.{k}", v)
        log(f"controls seed {seed} {model} {P}: fit {f['fit_acc']:.4f} seen {f['seen_acc']:.4f} unseen {f['unseen_acc']:.4f} "
            f"fresh {f['unseen_fresh_acc']:.4f} ({fit_s:.0f} s)")
        flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ stage (iii): oracle-initialised stability
def write_oracle(tm, raw: np.ndarray, weights: np.ndarray) -> None:
    """Allocate and initialise exactly as the first fit() would, then overwrite the TA state and the clause weights
    by direct host-to-device copies and clear `first` (so the next fit() neither re-allocates nor re-initialises)."""
    import pycuda.driver as cuda

    if not tm.first:
        raise RuntimeError("the oracle is written into a fresh machine only")
    tm.number_of_outputs = 1  # what TsetlinMachine.fit sets before _fit
    tm.max_y = None
    tm.min_y = None
    tm.allocate_gpu_memory()
    tm.initialize_weights_and_ta_states()
    tm.first = False
    need = int(tm.number_of_clauses) * int(tm.number_of_ta_teams) * int(tm.number_of_literal_chunks_per_leaf) \
        * int(tm.number_of_state_bits)
    raw = np.ascontiguousarray(raw, dtype=np.uint32)
    weights = np.ascontiguousarray(weights, dtype=np.int32)
    if raw.size != need or weights.size != int(tm.number_of_outputs) * int(tm.number_of_clauses):
        raise ValueError(f"state words {raw.size} (need {need}), weights {weights.size}")
    cuda.memcpy_htod(tm.ta_state_hierarchy_gpu, raw)
    cuda.memcpy_htod(tm.clause_weights_gpu, weights)
    cuda.Context.synchronize()


def read_back(tm) -> tuple[np.ndarray, np.ndarray]:
    import pycuda.driver as cuda

    raw = np.empty(int(tm.number_of_clauses) * int(tm.number_of_ta_teams) * int(tm.number_of_literal_chunks_per_leaf)
                   * int(tm.number_of_state_bits), dtype=np.uint32)
    cuda.memcpy_dtoh(raw, tm.ta_state_hierarchy_gpu)
    w = np.empty(int(tm.number_of_outputs) * int(tm.number_of_clauses), dtype=np.int32)
    cuda.memcpy_dtoh(w, tm.clause_weights_gpu)
    return raw, w


def set_state_audit(htm, tm) -> dict:
    src = inspect.getsource(htm)
    attrs = {k: int(getattr(tm, k)) for k in ("number_of_clauses", "number_of_ta_teams", "number_of_literal_chunks_per_leaf",
                                              "number_of_state_bits")}
    attrs["number_of_outputs"] = int(getattr(tm, "number_of_outputs", 1))  # set by fit(); a TsetlinMachine has 1
    out = HG.set_state_allocation(src, attrs)
    out["tm_py_sha256"] = hashing.sha256_bytes(src.encode("utf-8"))
    return out


def package_vs_reference(tm, orc: HG.OracleHierarchy, inc: np.ndarray, P: dict, S: dict) -> dict:
    """Package class sums on every bag against the numpy reference evaluated on `inc` (exact comparison; skipped
    for HEAD's and_group_normalization without binary_inference, which the reference does not model)."""
    binary = bool(P.get("binary_inference", 0))
    if bool(P.get("and_group_normalization", 0)) and not binary:
        return {"compared": False, "reason": "and_group_normalization without binary_inference is not modelled"}
    T = float(P["T"])
    out = {"compared": True}
    for name in SETS:
        ref, _ = HG.reference_scores(inc, orc.weights, S[name].X, orc.alternatives, orc.n_concepts, orc.n_features,
                                     orc.append_negated, binary)
        if np.abs(ref).max() >= 2 ** 24:
            out[name] = None
            continue
        ref = ref.astype(np.float64)
        if not bool(P.get("no_clipping", 0)):
            ref = np.clip(ref, -T, T)
        out[name] = bool(np.array_equal(scores(tm, S[name].X), ref))
    return out


def stage_oracle(cfg: dict, args) -> int:
    htm, tag = load_package()
    seed = int(args.world_seed)
    S = world_sets(cfg, seed)
    world = S["world"]
    V = int(world.vocab)
    rel, T = tuned_config(cfg, tag)
    O = cfg["params"]["oracle"]
    models = args.models.split(",") if args.models else list(O["models"])
    audited = False
    flush()
    for model in models:
        spec = T["params"]["models"][model]
        if spec["kind"] != "hier":
            raise SystemExit(f"{model} is not hierarchical")
        P0 = model_params(spec)
        neg = bool(P0["append_negated"])
        bits = int(P0.get("state_bits", 8))
        if int(P0["n_concepts"]) != world.n_groups:
            raise SystemExit(f"{model}: {P0['n_concepts']} concept nodes for {world.n_groups} groups")
        design = O["neg_design"]["negated" if neg else "positive"]
        orc = HG.oracle_hierarchy(world.groups, V, int(P0["clauses"]), int(P0["alternatives"]), neg, design)
        binary = bool(P0.get("binary_inference", 0))
        # the oracle's logic on the data, before any GPU work (numpy reference of the package's inference)
        for name in SETS:
            ref, out = HG.reference_scores(orc.inc, orc.weights, S[name].X, orc.alternatives, orc.n_concepts, V, neg, binary)
            check(f"seed{seed}.{model}.oracle_reference_predicts_{name}", bool(np.array_equal(ref >= 0, S[name].y.astype(bool))),
                  accuracy=float(np.mean((ref >= 0) == S[name].y.astype(bool))))
            for role in ("bias", "never"):
                idx = [i for i, (r, _) in enumerate(orc.roles) if r == role]
                if idx:
                    want = 1 if role == "bias" else 0
                    check(f"seed{seed}.{model}.oracle_{role}_clauses_output_{want}_on_{name}", bool(np.all(out[:, idx] == want)))
        M(f"oracle.{tag}.seed{seed}.{model}.design", {"design": orc.design, "roles": [r for r, _ in orc.roles][:8],
                                                       "spare_alternatives": "cycle the group's words (alternative a holds word a mod G)"},
          label="derived")
        for depth in O["depths"]:
            states = HG.oracle_states(orc.inc, bits, depth)
            raw = HG.encode_ta_state(states, bits)
            for s in O["s_values"]:
                P = {**P0, "s": float(s)}
                name = f"{model}.{depth}.s{float(s):g}"
                if args.replay_check and not audited:
                    replay_check(htm, tag, f"oracle.{name}", "hier", P, S, seed,
                                 prepare=lambda t: write_oracle(t, raw, orc.weights))
                tm = make_model(htm, tag, "hier", V, P, seed)
                if not audited:
                    aud = set_state_audit(htm, tm)
                    M(f"oracle.{tag}.set_state_audit", aud, label="derived",
                      note="parsed from the installed package source (read only) with this machine's sizes")
                    log(f"set_state audit ({tag}): {json.dumps(aud)}")
                    audited = True
                write_oracle(tm, raw, orc.weights)
                rb_raw, rb_w = read_back(tm)
                inc_now = E.pkg_included(tm)
                rb = {"raw_words_equal": bool(np.array_equal(rb_raw, raw)), "weights_equal": bool(np.array_equal(rb_w, orc.weights)),
                      "includes_equal": bool(np.array_equal(inc_now, orc.inc)),
                      "package_get_ta_states_equal": bool(np.array_equal(np.asarray(tm.get_ta_states()), states))}
                pv = package_vs_reference(tm, orc, inc_now, P, S)
                ev0 = eval_sets(tm, S)
                pmin = float(O["predict_min"])
                ok = all(rb.values()) and all(v is not False for k, v in pv.items() if k in SETS) \
                    and all(ev0[f"{n}_acc"] >= pmin for n in SETS)
                check(f"oracle.{tag}.seed{seed}.{name}.written_and_verified", ok, read_back=rb, package_vs_reference=pv,
                      accuracy_epoch0={n: ev0[f"{n}_acc"] for n in SETS}, predict_min=pmin)
                cell = {"stage": "oracle", "tag": tag, "world_seed": seed, "model": model, "depth": depth, "s": float(s),
                        "tuned_config": rel, "design": orc.design, "read_back": rb, "package_vs_reference_epoch0": pv,
                        "verified": ok}
                if not ok:
                    cell["skipped"] = "oracle write not verified: training not run (a failed write is not a stability result)"
                    append_row(cell)
                    flush()
                    del tm
                    continue
                c, tm = train_curve(htm, tag, name, "hier", P, S, seed, int(O["epochs"]), int(O["eval_every"]),
                                    float(O["wall_cap_s"]), oracle=orc, tm=tm)
                pv_end = package_vs_reference(tm, orc, E.pkg_included(tm), P, S)
                check(f"oracle.{tag}.seed{seed}.{name}.package_equals_reference_at_end",
                      all(v is not False for k, v in pv_end.items() if k in SETS), detail=pv_end)
                w_end = read_back(tm)[1]
                check(f"oracle.{tag}.seed{seed}.{name}.weights_unchanged", bool(np.array_equal(w_end, orc.weights)))
                rule = O["stable_rule"]
                f = c["final"]
                stable = (c["epochs_run"] == int(O["epochs"]) and f["unseen_acc"] >= float(rule["min_unseen_acc"])
                          and f.get("pos_node_jaccard_mean", 0.0) >= float(rule["min_pos_node_jaccard_mean"]))
                c.update(cell)
                c["package_vs_reference_end"] = pv_end
                c["stable"] = bool(stable)
                append_row(c)
                pre = f"oracle.{tag}.seed{seed}.{name}"
                M(f"{pre}.final", {k: f[k] for k in ("epoch", "fit_acc", "seen_acc", "unseen_acc", "unseen_pos_acc",
                                                     "unseen_neg_acc", "pos_node_jaccard_mean", "pos_node_share_exact",
                                                     "dm_node_jaccard_mean", "clauses_exact_share") if k in f})
                M(f"{pre}.stable", bool(stable), label="derived", rule=rule["text"])
                log(f"oracle {name}: stable {stable} (unseen {f['unseen_acc']:.4f}, Jnode+ {f.get('pos_node_jaccard_mean', -1):.3f})")
                flush()
                del tm
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ stage (iv): epochs to threshold
def stage_threshold(cfg: dict, args) -> int:
    htm, tag = load_package()
    Th = cfg["params"]["threshold"]
    seed = int(args.world_seed)
    if seed != int(Th["world_seed"]):
        raise SystemExit(f"one instance: world seed {Th['world_seed']} only")
    sel = best_hier_config(cfg)
    M("threshold.selection", sel, label="derived", rule="highest val_acc_at_chosen_epoch among the hierarchical models of both tuned configs")
    best = sel["best"]
    if best["tag"] != tag:
        raise SystemExit(f"the best tuned hierarchical configuration is on {best['tag']}: run this stage under its venv")
    S = world_sets(cfg, seed)
    rel, T = tuned_config(cfg, tag)
    spec = T["params"]["models"][best["model"]]
    P = model_params(spec)
    flush()
    cell, tm = train_curve(htm, tag, best["model"], spec["kind"], P, S, seed, int(Th["max_epochs"]), int(Th["eval_every"]),
                           float(Th["wall_cap_s"]), stop_at=float(Th["target_fit_acc"]))
    del tm
    check_tuning_reproduction(cfg, f"threshold.{tag}.seed{seed}.{best['model']}", seed, cell,
                              T["tuning"]["chosen"].get(best["model"]), S)
    hit = [r["epoch"] for r in cell["curve"] if r["fit_acc"] >= float(Th["target_fit_acc"])]
    cell.update({"stage": "threshold", "tag": tag, "world_seed": seed, "model": best["model"], "tuned_config": rel,
                 "selection": sel, "epochs_to_target": hit[0] if hit else None, "target_fit_acc": float(Th["target_fit_acc"])})
    append_row(cell)
    M(f"threshold.{tag}.seed{seed}.{best['model']}.epochs_to_target", hit[0] if hit else None,
      note=f"first evaluated epoch (every {Th['eval_every']}) with fit accuracy >= {Th['target_fit_acc']}; None = not reached")
    M(f"threshold.{tag}.seed{seed}.{best['model']}.stop_reason", cell["stop_reason"])
    M(f"threshold.{tag}.seed{seed}.{best['model']}.final", summary_keys(cell))
    M(f"threshold.{tag}.seed{seed}.{best['model']}.epochs_run", cell["epochs_run"])
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ stage (vi): Granmo's demos
def git_show(commit: str, path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REF_REPO), "show", f"{commit}:{path}"], capture_output=True, text=True,
                              check=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        log(f"git show {commit}:{path} failed: {exc}")
        return None


def demo_source_check(tag: str, demo: str, C: dict) -> None:
    commit = C85 if tag == "c85" else HD.PIN
    src = git_show(commit, C["source"])
    if src is None:
        check(f"demos.{tag}.{demo}.source_readable", False)
        return
    check(f"demos.{tag}.{demo}.source_sha256", hashing.sha256_bytes(src.encode("utf-8")) == C["source_sha256"],
          got=hashing.sha256_bytes(src.encode("utf-8")), want=C["source_sha256"])
    parsed = HG.parse_argparse_defaults(src)
    check(f"demos.{tag}.{demo}.argparse_defaults_equal_config", parsed == C["defaults"], parsed=parsed, config=C["defaults"])
    if "runs" in parsed:
        check(f"demos.{tag}.{demo}.runs_equal_the_default", int(C["runs"]) == int(parsed["runs"]), runs=C["runs"])


def switch_off_kwargs(tag: str) -> dict:
    """The demos at HEAD pass their four switch flags, all False by default; c85ca4e7 has no switches."""
    return {"binary_inference": False, "constant_update_p": False, "and_group_normalization": False,
            "no_clipping": False} if tag == "head" else {}


def run_demo_variant(htm, tag: str, demo: str, C: dict, variant: str) -> dict:
    d = C["defaults"]
    clauses = int(np.prod([int(d[f]) for f in C["machine_clauses"][variant]]))
    rs = np.random.RandomState(int(C["machine_seed_seed"]))
    if demo == "andxor":
        irr, n_ands, A = int(d["number_of_irrelevant_features"]), int(d["number_of_ands"]), int(d["number_of_alternatives"])
        Xtr, ytr, ytr_clean, Xte, yte = HD.andxor_data(int(C["data_seed"]), n_ands, irr, int(d["number_of_training_examples"]),
                                                       int(d["number_of_testing_examples"]), float(d["noise"]))
        check(f"demos.{tag}.andxor.labels_follow_rule", bool(np.array_equal(HD.xor_rule(Xtr, n_ands, irr), ytr_clean))
              and bool(np.array_equal(HD.xor_rule(Xte, n_ands, irr), yte)))
        w = 2 + irr
        hier = ((htm.AND_GROUP, w), (htm.OR_ALTERNATIVES, A), (htm.AND_GROUP, n_ands)) if variant == "hier" \
            else ((htm.AND_GROUP, w * n_ands), (htm.OR_ALTERNATIVES, A))
        bits = int(d["number_of_state_bits"])
    else:
        E_, A, copies = int(d["number_of_elements"]), int(d["number_of_alternatives"]), int(d["number_of_copies"])
        Xtr, ytr, ytr_clean, Xte, yte = HG.multiconcept_data(int(C["data_seed"]), E_, int(d["number_of_examples"]), float(d["noise"]))
        check(f"demos.{tag}.multiconcept.labels_follow_rule", bool(np.array_equal(HG.multiconcept_rule(Xtr, E_), ytr_clean))
              and bool(np.array_equal(HG.multiconcept_rule(Xte, E_), yte)))
        hier = ((htm.AND_GROUP, 2 * E_), (htm.OR_ALTERNATIVES, A), (htm.AND_ALTERNATIVES, copies)) if variant == "hier" \
            else ((htm.AND_GROUP, 2 * E_), (htm.AND_GROUP, 1))
        bits = int(C.get("state_bits") or d["number_of_state_bits"])
    data_sha = {"Xtr": E.arr_sha(Xtr), "ytr": E.arr_sha(np.asarray(ytr, np.uint32)), "Xte": E.arr_sha(Xte), "yte": E.arr_sha(yte),
                "train_noise_rate": float(np.mean(ytr != ytr_clean)), "train_pos_rate": float(np.mean(ytr_clean)),
                "test_pos_rate": float(np.mean(yte))}
    runs = []
    t_var = time.time()
    capped = False
    for r in range(int(C["runs"])):
        mseed = int(rs.randint(*C["machine_seed_randint"]))
        kw = switch_off_kwargs(tag)
        if demo == "andxor":
            tm = htm.TsetlinMachine(clauses, int(d["T"]), float(d["s"]), number_of_state_bits=bits, boost_true_positive_feedback=0,
                                    hierarchy_structure=hier, seed=mseed, grid=E.GRID, block=E.BLOCK, **kw)
        else:
            np.random.seed(mseed)  # MultiClassTsetlinMachine.fit samples and shuffles with the global generator
            tm = htm.MultiClassTsetlinMachine(clauses, int(d["T"]), float(d["s"]), seed=mseed, number_of_state_bits=bits,
                                              boost_true_positive_feedback=0, append_negated=False, hierarchy_structure=hier,
                                              grid=E.GRID, block=E.BLOCK, **kw)
        curve = []
        for e in range(int(d["epochs"])):
            te = time.time()
            tm.fit(Xtr, ytr)
            ep_s = time.time() - te
            acc = float(np.mean(np.asarray(tm.predict(Xte)).reshape(-1) == yte))
            curve.append({"epoch": e + 1, "test_acc": acc, "epoch_s": ep_s})
            if (e + 1) % max(1, int(d["epochs"]) // 20) == 0 or e == 0 or e + 1 == int(d["epochs"]):
                log(f"demo {tag} {demo} {variant} run {r + 1}/{C['runs']} epoch {e + 1}: test {acc:.4f} ({ep_s:.2f} s)")
            if time.time() - t_var > float(C["wall_cap_s"]):
                capped = True
                break
        thr = float(C["learn_threshold"])
        first = next((row["epoch"] for row in curve if row["test_acc"] >= thr), None)
        runs.append({"run": r, "machine_seed": mseed, "curve": curve, "final_test_acc": curve[-1]["test_acc"],
                     "best_test_acc": max(row["test_acc"] for row in curve), "first_epoch_at_threshold": first,
                     "epochs_run": len(curve)})
        del tm
        if capped:
            log(f"demo {tag} {demo} {variant}: wall cap {C['wall_cap_s']} s reached in run {r + 1}")
            break
    finals = [x["final_test_acc"] for x in runs if x["epochs_run"] == int(d["epochs"])]
    return {"stage": "demos", "tag": tag, "demo": demo, "variant": variant, "machine_clauses": clauses,
            "hierarchy": [list(h) for h in hier], "state_bits": bits, "defaults": d, "runs_planned": int(C["runs"]),
            "runs": runs, "wall_capped": capped, "data": data_sha,
            "summary_final_complete_runs": HG.summarize_runs(finals),
            "summary_best": HG.summarize_runs([x["best_test_acc"] for x in runs])}


def stage_demos(cfg: dict, args) -> int:
    htm, tag = load_package()
    Dm = cfg["params"]["demos"][tag]
    demos = args.demos.split(",") if args.demos else ["andxor", "multiconcept"]
    flush()
    for demo in demos:
        C = Dm[demo]
        demo_source_check(tag, demo, C)
        flush()
        for variant in C["variants"]:
            cell = run_demo_variant(htm, tag, demo, C, variant)
            append_row(cell)
            pre = f"demos.{tag}.{demo}.{variant}"
            M(f"{pre}.final_test_acc", cell["summary_final_complete_runs"],
              note="over the runs that completed the default epochs (the c85ca4e7 ANDOfXORs demo prints this mean as 'Average Accuracy')")
            M(f"{pre}.best_test_acc", cell["summary_best"])
            M(f"{pre}.runs_completed", sum(1 for x in cell["runs"] if x["epochs_run"] == int(C["defaults"]["epochs"])))
            M(f"{pre}.wall_capped", cell["wall_capped"])
            flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ selftest (tiny, GPU): the runner on the installed commit
TINY_WORLD = {"vocab": 64, "n_groups": 3, "group_size": 4, "n_distractors": 5, "train_fraction": 0.6, "extra_probs": [1.0],
              "val_fraction": 0.2, "n_fit": 240, "n_val": 120}
TINY_DEMO = {"andxor": {"epochs": 2, "number_of_training_examples": 500, "number_of_testing_examples": 200},
             "multiconcept": {"epochs": 2, "number_of_examples": 300, "number_of_alternatives": 8}}
ROUND_TRIP_CASES = ((64, 4, "deep"), (60, 5, "deep"), (60, 5, "boundary"))  # vocab 60: a partly filled last chunk


def tiny_config(cfg: dict, tag: str, out: Path) -> Path:
    """The config on a tiny world with a tiny tuned config of commit `tag` (the four package models at toy sizes,
    HEAD's switches as in its tuned config at HEAD; the five controls at toy sizes), written into `out` and returned
    as a path. The reference seed is 0: no stage compares with the H-AO tuning data. Used by --stage selftest and the
    tests."""
    c = json.loads(json.dumps(cfg))
    P = c["params"]
    P.update({"world": dict(TINY_WORLD), "n_seen": 120, "n_unseen_fresh": 120})
    P["reference"]["seed"] = 0
    P["valiant"].update({"max_or": 4, "n_pos_curve": [10, "all"]})
    P["oracle"].update({"s_values": [4], "epochs": 2, "eval_every": 1})
    P["threshold"].update({"eval_every": 1, "max_epochs": 3, "wall_cap_s": 600})

    def sw(b: int, cp: int) -> dict:
        return {"binary_inference": b, "constant_update_p": cp, "and_group_normalization": 0, "no_clipping": 0} if tag == "head" else {}
    base = {"T": 100000, "state_bits": 8, "weighted_clauses": False}
    hier = {**base, "kind": "hier", "clauses": 6, "alternatives": 4, "n_concepts": 3}
    flat = {**base, "kind": "flat", "clauses": 16, "alternatives": 1, "n_concepts": 1}
    models = {"htm_hier_neg": {**hier, "append_negated": True, "boost": 0, **sw(0, 0), "epochs": 3, "grid": [{"s": 4.0}]},
              "htm_flat_neg": {**flat, "append_negated": True, "boost": 0, **sw(0, 1), "epochs": 2, "grid": [{"s": 4.0}]},
              "htm_hier_pos": {**hier, "append_negated": False, "boost": 1, **sw(1, 1), "epochs": 2, "grid": [{"s": 10.0}]},
              "htm_flat_pos": {**flat, "append_negated": False, "boost": 1, **sw(0, 1), "epochs": 2, "grid": [{"s": 10.0}]},
              "tmu_tm": {"kind": "tmu_tm", "clauses": 16, "T": 10, "epochs": 2, "grid": [{"s": 5.0}, {"s": 20.0}]},
              "tmu_ctm": {"kind": "tmu_ctm", "clauses": 16, "T": 10, "epochs": 2, "grid": [{"s": 5.0}, {"s": 20.0}]},
              "lr": {"kind": "lr", "c_grid": [0.1, 1.0], "folds": 2, "max_iter": 200},
              "gbt": {"kind": "gbt", "max_iter": 10, "learning_rate": 0.1, "max_leaf_nodes": 7},
              "mlp": {"kind": "mlp", "hidden": [8], "alpha": 0.0001, "max_iter": 20, "batch_size": 50}}
    chosen = {m: {"val_acc_at_chosen_epoch": v, "train_acc_at_chosen_epoch": v, "tuning_config": m,
                  "chosen_epoch": models[m]["epochs"]}
              for m, v in (("htm_hier_neg", 0.7), ("htm_flat_neg", 0.8), ("htm_hier_pos", 0.5), ("htm_flat_pos", 0.5))}
    tuned = {"package": {"commit": HD.PIN if tag == "head" else C85},
             "params": {"world": dict(TINY_WORLD), "models": models}, "tuning": {"chosen": chosen}}
    out.mkdir(parents=True, exist_ok=True)
    tp = out / f"tiny_tuned_{tag}.json"
    tp.write_text(json.dumps(tuned), encoding="utf-8")
    P["tuned_configs"] = {tag: str(tp)}
    P["controls"]["tuned_config"] = tag
    cp = out / f"tiny_cfg_{tag}.json"
    cp.write_text(json.dumps(c), encoding="utf-8")
    return cp


def oracle_round_trip(htm, tag: str, vocab: int, alternatives: int, depth: str, neg: bool, switches: dict | None = None,
                      seed: int = 9) -> dict:
    """The oracle write on a tiny package instance (3 groups of 4 words): read-back of the raw words, the weights,
    the includes and the package's get_ta_states; package class sums equal to the numpy reference; exact
    predictions; then one fit() at s = 1e9 keeps `first` cleared, moves TA states and keeps the includes (deep:
    exactly; boundary: every oracle include survives, an excluded TA at 127 may cross to 128). Returns
    {"checks": {name: bool}, "info": {...}}; used by --stage selftest and the GPU tests."""
    w = H.make_world(seed, vocab=int(vocab), n_groups=3, group_size=4, train_fraction=0.6)
    fm, vm = H.carve_combos(w, w.train_combo, 0.2, 5)
    kw = {"n_distractors": 5, "extra_probs": (1.0,)}
    S = {"world": w, "fit": H.sample_bags(w, 200, fm, 3, **kw), "seen": H.sample_bags(w, 100, fm, 21, **kw),
         "unseen": H.sample_bags(w, 100, vm, 4, **kw), "unseen_fresh": H.sample_bags(w, 100, vm, 22, **kw)}
    P = {"clauses": 6, "alternatives": int(alternatives), "n_concepts": 3, "T": 100000, "s": 1e9, "state_bits": 8,
         "boost": 0, "append_negated": bool(neg), "weighted_clauses": False, **(switches or {})}
    orc = HG.oracle_hierarchy(w.groups, w.vocab, 6, int(alternatives), bool(neg))
    states = HG.oracle_states(orc.inc, 8, depth)
    raw = HG.encode_ta_state(states, 8)
    tm = make_model(htm, tag, "hier", w.vocab, P, seed)
    aud = set_state_audit(htm, tm)
    ch = {"audit_under_allocates_by_chunks_per_leaf": bool(aud["under_allocates"] and not aud["sets_first"]
          and aud["needed_over_alloc"] == int(tm.number_of_literal_chunks_per_leaf) == HG.n_chunks(orc.literals))}
    write_oracle(tm, raw, orc.weights)
    rb_raw, rb_w = read_back(tm)
    pv = package_vs_reference(tm, orc, orc.inc, P, S)
    ch.update({"first_cleared": tm.first is False, "raw_words_equal": bool(np.array_equal(rb_raw, raw)),
               "weights_equal": bool(np.array_equal(rb_w, orc.weights)),
               "includes_equal": bool(np.array_equal(E.pkg_included(tm), orc.inc)),
               "package_get_ta_states_equal": bool(np.array_equal(np.asarray(tm.get_ta_states()), states)),
               "class_sums_equal_reference": bool(pv["compared"] and all(pv[n] for n in SETS)),
               "predictions_exact": all(bool(np.array_equal(np.asarray(tm.predict(np.ascontiguousarray(S[n].X, dtype=np.uint32)))
                                                             .reshape(-1).astype(bool), S[n].y.astype(bool))) for n in SETS)})
    tm.fit(np.ascontiguousarray(S["fit"].X, dtype=np.uint32), np.ascontiguousarray(S["fit"].y, dtype=np.uint32))
    inc = E.pkg_included(tm)
    ch.update({"fit_keeps_first_cleared": tm.first is False, "fit_moved_ta_states": not np.array_equal(read_back(tm)[0], raw),
               "fit_keeps_includes": bool(np.array_equal(inc, orc.inc)) if depth == "deep" else bool(np.all(inc[orc.inc]))})
    info = {"vocab": int(vocab), "alternatives": int(alternatives), "depth": depth, "append_negated": bool(neg),
            "switches": switches or {}, "literals": orc.literals, "chunks_per_leaf": int(tm.number_of_literal_chunks_per_leaf),
            "last_chunk_partly_filled": orc.literals % 32 != 0, "includes_added_by_fit": int((inc & ~orc.inc).sum()),
            "clauses_exact_share_after_fit": HG.node_recovery(inc, orc)["clauses_exact_share"], "audit": aud}
    del tm
    return {"checks": ch, "info": info}


def stage_selftest(cfg: dict, args) -> int:
    htm, tag = load_package()
    if not check("selftest.package_commit_is_expected", tag == args.expect_tag, detected=tag, expected=args.expect_tag):
        return 1
    for vocab, alts, depth in ROUND_TRIP_CASES:
        for neg in (True, False):
            sw = {"binary_inference": 1, "constant_update_p": 1} if (tag == "head" and not neg) else {}
            r = oracle_round_trip(htm, tag, vocab, alts, depth, neg, sw)
            check(f"selftest.round_trip.V{vocab}.A{alts}.{depth}.{'neg' if neg else 'pos'}", all(r["checks"].values()), **r)
    flush()
    base = out_dir()
    sub = base / "selftest"
    tiny = json.loads(tiny_config(cfg, tag, sub).read_text(encoding="utf-8"))
    ns = argparse.Namespace(world_seed=100, models=None, replay_check=True, demos=None)
    _OUT[0] = sub  # the tiny stages' cells and hierarchy files go to <run dir>/selftest/, never into the run's cells.jsonl
    try:
        for stage in ("oracle", "seen_unseen", "pospair", "threshold"):
            n0 = len(FAILED)
            if stage == "oracle":
                stage_oracle(tiny, ns)
            elif stage == "threshold":
                stage_threshold(tiny, ns)
            else:
                stage_models(tiny, ns, stage)
            check(f"selftest.stage.{stage}", len(FAILED) == n0, new_failures=FAILED[n0:])
        rows = [json.loads(x) for x in (sub / "cells.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        orows = [r for r in rows if r["stage"] == "oracle"]
        check("selftest.stage.oracle.rows_verified_and_exact_at_epoch0", len(orows) == 4 and all(
            r["verified"] and r["curve"][0]["unseen_acc"] == 1.0 and r["curve"][0]["pos_node_jaccard_mean"] == 1.0 for r in orows),
            n_rows=len(orows))
        for demo, C in cfg["params"]["demos"][tag].items():
            Ct = {**C, "defaults": {**C["defaults"], **TINY_DEMO[demo]}, "runs": min(2, int(C["runs"])), "wall_cap_s": 600}
            for variant in C["variants"]:
                a, b = (run_demo_variant(htm, tag, demo, Ct, variant) for _ in range(2))
                curves = [[[e["test_acc"] for e in r["curve"]] for r in x["runs"]] for x in (a, b)]
                check(f"selftest.demo.{demo}.{variant}.two_seeded_runs_equal", curves[0] == curves[1]
                      and [r["machine_seed"] for r in a["runs"]] == [r["machine_seed"] for r in b["runs"]]
                      and len(a["runs"]) == Ct["runs"] and all(len(c) == 2 for c in curves[0]), curves=curves[0])
            demo_source_check(tag, demo, C)
    finally:
        _OUT[0] = base
    M("selftest.passed", not FAILED, label="derived")
    M("wall_s", time.time() - T0)
    flush()
    log(f"selftest ({tag}): {'PASSED' if not FAILED else 'FAILED ' + json.dumps(FAILED)}")
    return 1 if FAILED else 0


# ------------------------------------------------------------------ stage (v): Valiant elimination
def stage_valiant(cfg: dict, args) -> int:
    seed = int(args.world_seed)
    S = world_sets(cfg, seed)
    world = S["world"]
    Vc = cfg["params"]["valiant"]
    fit = S["fit"]
    truth = [set(int(w) for w in g) for g in world.groups]
    group_words = set().union(*truth)
    flush()
    for npos in Vc["n_pos_curve"]:
        n_arg = None if npos == "all" else int(npos)
        t0 = time.time()
        model = HG.valiant_fit(fit.indptr, fit.indices, fit.y, world.vocab, float(Vc["min_freq"]), int(Vc["max_or"]),
                               int(Vc["max_candidates"]), n_pos=n_arg)
        fit_s = time.time() - t0
        row = {"stage": "valiant", "world_seed": seed, "n_pos_arg": npos, "fit_s": fit_s,
               **{k: model[k] for k in ("n_pos", "min_freq", "max_or", "kept", "kept_freq", "n_candidates",
                                         "n_distinct_positive_masks", "per_size", "n_survivors", "minimal")}}
        for name in SETS:
            b = S[name]
            p = HG.valiant_predict(model, b.indptr, b.indices)
            y = b.y.astype(bool)
            row[f"{name}_acc"] = float(np.mean(p == y))
            row[f"{name}_pos_acc"] = float(np.mean(p[y]))
            row[f"{name}_neg_acc"] = float(np.mean(~p[~y]))
        p_all = HG.valiant_predict(model, S["unseen"].indptr, S["unseen"].indices, use="all")
        p_min = HG.valiant_predict(model, S["unseen"].indptr, S["unseen"].indices, use="minimal")
        check(f"valiant.seed{seed}.n{npos}.minimal_predicts_like_all", bool(np.array_equal(p_all, p_min)))
        minimal = [set(m) for m in model["minimal"]]
        kept = set(model["kept"])
        row.update({"kept_group_words": len(kept & group_words), "kept_distractors": len(kept - group_words),
                    "true_groups_recovered_exactly": [bool(t in minimal) for t in truth],
                    "n_true_groups_recovered": int(sum(t in minimal for t in truth)),
                    "n_minimal": len(minimal), "n_spurious_minimal": int(sum(m not in truth for m in minimal)),
                    "recovery": H.or_group_recovery(minimal, world.groups) if minimal else {"mean_jaccard": 0.0}})
        append_row(row)
        pre = f"valiant.seed{seed}.n{npos}"
        for k in ("n_pos", "fit_acc", "seen_acc", "unseen_acc", "unseen_pos_acc", "unseen_neg_acc", "unseen_fresh_acc",
                  "n_true_groups_recovered",
                  "n_minimal", "n_spurious_minimal", "kept_group_words", "kept_distractors", "n_candidates", "n_survivors"):
            M(f"{pre}.{k}", row[k])
        M(f"{pre}.recovery_mean_jaccard", row["recovery"]["mean_jaccard"])
        log(f"valiant seed {seed} n_pos {model['n_pos']}: kept {len(kept)} ({row['kept_group_words']} group words), "
            f"survivors {model['n_survivors']}, minimal {len(minimal)} ({row['n_true_groups_recovered']} true groups, "
            f"{row['n_spurious_minimal']} spurious); fit {row['fit_acc']:.4f} seen {row['seen_acc']:.4f} "
            f"unseen {row['unseen_acc']:.4f} fresh {row['unseen_fresh_acc']:.4f} ({fit_s:.1f} s)")
        flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ report (reading notes of the config)
def read_rows(run_dirs: list[str]) -> list[dict]:
    rows = []
    for d in run_dirs:
        f = Path(d) / "cells.jsonl"
        if not f.exists():
            log(f"report: no cells.jsonl in {d}")
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_run"] = str(d)
                rows.append(r)
    return rows


def stage_report(cfg: dict, args) -> int:
    R = cfg["reading_notes"]
    rows = read_rows([d for d in (args.runs or "").split(",") if d])
    M("report.runs", sorted({r["_run"] for r in rows}))
    out: dict = {}
    # (vi) demos
    thr = float(R["demos"]["pass_mean_final_test_acc"])
    for r in (x for x in rows if x.get("stage") == "demos"):
        sm = r["summary_final_complete_runs"]
        out.setdefault("demos", {})[f"{r['tag']}.{r['demo']}.{r['variant']}"] = {
            "mean_final_test_acc": sm.get("mean"), "runs": sm.get("n"), "runs_planned": r["runs_planned"],
            "passes": bool(sm.get("n") == r["runs_planned"] and (sm.get("mean") or 0.0) >= thr)}
    # (ii) and (i)
    Rs = R["seen_unseen"]
    ref_seed = int(cfg["params"]["reference"]["seed"])
    for r in (x for x in rows if x.get("stage") in ("seen_unseen", "pospair", "controls")):
        f = r["final"]
        flags = {"memorises_combinations": f["gap_seen_minus_unseen"] >= float(Rs["combination_gap"]),
                 "memorises_bags": f["gap_fit_minus_seen"] >= float(Rs["bag_gap"]),
                 "generalises_to_unseen_combinations": f["unseen_acc"] >= float(Rs["generalise_min_unseen_acc"])
                 and abs(f["gap_seen_minus_unseen"]) < float(Rs["generalise_max_gap"]),
                 "memorises_combinations_fresh": f["gap_seen_minus_unseen_fresh"] >= float(Rs["combination_gap"]),
                 "generalises_to_unseen_combinations_fresh": f["unseen_fresh_acc"] >= float(Rs["generalise_min_unseen_acc"])
                 and abs(f["gap_seen_minus_unseen_fresh"]) < float(Rs["generalise_max_gap"]),
                 "unseen_is_tuning_validation": int(r["world_seed"]) == ref_seed}
        out.setdefault(r["stage"], {})[f"{r['tag']}.seed{r['world_seed']}.{r['model']}"] = {**summary_keys(r), **flags}
    Rp = R["pospair"]
    pp = out.get("pospair", {})
    for key in list(pp):
        if key.endswith(".htm_hier_pos"):
            other = key[: -len("htm_hier_pos")] + "htm_flat_pos"
            if other in pp:
                d_un = pp[key]["unseen_acc"] - pp[other]["unseen_acc"]
                reading = ("HIERARCHY HELPS WHERE FLAT CANNOT EXPRESS COMPACTLY" if d_un >= float(Rp["helps_min_diff"])
                           and pp[key]["unseen_acc"] >= float(Rp["helps_min_hier_unseen_acc"])
                           else "NEITHER LEARNS" if max(pp[key]["fit_acc"], pp[other]["fit_acc"]) < float(Rp["neither_max_fit_acc"])
                           else "NO HIERARCHY ADVANTAGE")
                out.setdefault("pospair_pairs", {})[key[: -len(".htm_hier_pos")]] = {"hier_minus_flat_unseen": d_un, "reading": reading}
    # (iii) oracle
    Ro = cfg["params"]["oracle"]["stable_rule"]
    stab: dict = {}
    for r in (x for x in rows if x.get("stage") == "oracle"):
        k = f"{r['tag']}.{r['model']}.{r['depth']}.s{float(r['s']):g}"
        entry = {"verified": r.get("verified", False), "stable": r.get("stable"), "final": r.get("final", {}).get("unseen_acc"),
                 "pos_node_jaccard": r.get("final", {}).get("pos_node_jaccard_mean")}
        stab.setdefault(k, {})[f"seed{r['world_seed']}"] = entry
    out["oracle"] = stab
    trig = sorted(k for k, v in stab.items() if k.split(".")[2] == R["oracle"]["factorial_depth"]
                  and all(f"seed{s}" in v and v[f"seed{s}"]["stable"] for s in R["oracle"]["factorial_seeds"]))
    out["oracle_factorial_trigger"] = {"triggered": bool(trig), "stable_cells": trig, "rule": R["oracle"]["factorial_trigger"],
                                       "stable_rule": Ro["text"]}
    # (iv) threshold
    for r in (x for x in rows if x.get("stage") == "threshold"):
        out.setdefault("threshold", {})[f"{r['tag']}.seed{r['world_seed']}.{r['model']}"] = {
            "epochs_to_target": r["epochs_to_target"], "stop_reason": r["stop_reason"], "epochs_run": r["epochs_run"],
            "final": summary_keys(r)}
    # (v) Valiant
    Rv = R["valiant"]
    for r in (x for x in rows if x.get("stage") == "valiant" and x.get("n_pos_arg") == "all"):
        out.setdefault("valiant", {})[f"seed{r['world_seed']}"] = {
            "unseen_acc": r["unseen_acc"], "seen_acc": r["seen_acc"], "n_true_groups_recovered": r["n_true_groups_recovered"],
            "n_spurious_minimal": r["n_spurious_minimal"],
            "learns_the_concept": bool(r["unseen_acc"] >= float(Rv["learns_min_unseen_acc"])
                                       and r["n_true_groups_recovered"] == len(r["true_groups_recovered_exactly"]))}
    M("report.readings", out, label="derived", notes_written=R["written"])
    (out_dir() / "readings.json").write_text(json.dumps(jsonable(out), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(jsonable(out), indent=1, sort_keys=True)[:6000])
    flush()
    return 0


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["demos", "seen_unseen", "pospair", "oracle", "threshold", "valiant",
                                                      "controls", "selftest", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--world-seed", type=int, default=100)
    ap.add_argument("--models", default=None, help="subset of the stage's tuned models (comma separated)")
    ap.add_argument("--demos", default=None, help="demos: andxor,multiconcept (default both)")
    ap.add_argument("--runs", default=None, help="report: comma separated run dirs holding cells.jsonl")
    ap.add_argument("--replay-check", action="store_true", help="train twice (1 epoch, 1,000 fit bags), compare state hashes")
    ap.add_argument("--expect-tag", default=None, choices=sorted(set(COMMITS.values())),
                    help="selftest: the package commit this venv must hold (c85 or head)")
    args = ap.parse_args(argv)
    if args.stage == "selftest" and args.expect_tag is None:
        ap.error("--stage selftest needs --expect-tag")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    M("config_file", args.config, sha256=hashing.sha256_file(args.config))
    M("stage", args.stage)
    M("world_seed", int(args.world_seed))
    M("label", "EXPLORATORY (docs/13 A8 item 3; world seeds 100 and 101; not a result, docs/13 section 3)")
    try:
        if args.stage in ("seen_unseen", "pospair"):
            return stage_models(cfg, args, args.stage)
        if args.stage == "oracle":
            return stage_oracle(cfg, args)
        if args.stage == "threshold":
            return stage_threshold(cfg, args)
        if args.stage == "demos":
            return stage_demos(cfg, args)
        if args.stage == "valiant":
            return stage_valiant(cfg, args)
        if args.stage == "controls":
            return stage_controls(cfg, args)
        if args.stage == "selftest":
            return stage_selftest(cfg, args)
        return stage_report(cfg, args)
    finally:
        M("checks_failed", list(FAILED))
        flush()


if __name__ == "__main__":
    sys.exit(main())
