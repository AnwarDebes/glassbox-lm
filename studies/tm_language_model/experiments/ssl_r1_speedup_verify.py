"""Verification of the exact speedups O4, O10 and O12 of the R1 confirmatory Q_pred and features stages
(docs/13 A8 item 7; config experiments/cvtm/configs/ssl_r1_speedup_verify.json, whose pass rules were written before any
run of this file). Not a registered row: it decides whether experiments/ssl_r1_confirm.py may use each speedup.

Stages (each its own logged run through scripts/run_logged.py, GPU lock shared):
  features    V2: ssl_r1_confirm.features_core (the one-pass prefix fire counts and Q_pred count table, O10) on the A3
              seed-100 exploratory train run, checkpoints c0y, young_b77, epoch4, with the control study's config;
              rules F1 to F4 against the logged features run.
  qpred       V3 (--features-run V2): ssl_r1_confirm.qpred_core with O4 (T1GpuG), O10 (tables loaded from V2) and O12
              (young_b77 = c0y by state hash: copied); rules Q1 to Q6 against the logged Q_pred run.
  crosscheck  V4 (--features-run V2 --qpred-run V3): K1 every output of T1GpuG.run bit-identical to T1Gpu.run on the
              registered tables of c0y and epoch4 at the configured (alpha, beta0) points; K2 the loaded tables equal
              the registered count_table recomputed; K3 young_b77 scored by the registered qpred_checkpoint equals V3's
              copies; plus interleaved kernel timings.
Every rule is a check of the run: the run is clean (exit 0, failed_checks empty) exactly when every rule holds. The
WikiText-103 test split is never loaded (Q_pred reads the frozen validation targets; the stages assert no test access).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for _q in (ROOT / "src", EXPERIMENTS):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_confirm as S  # noqa: E402
import ssl_r1_controls as C  # noqa: E402  (imported, never edited)
import ssl_r1_pretext as R  # noqa: E402  (imported, never edited)
from tmlm.common import hashing  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import qpred_fast as QF  # noqa: E402

T0 = time.time()
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_speedup_verify.json"
FLAGS = ("exploratory", "confirmatory", "smoke", "verification")
FIT_TIMING = ("wall_s", "s_per_evaluation")


class Rec(R.Rec):
    """R.Rec with verification labels (not exploratory, not confirmatory)."""

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, np.bool_):
            value = bool(value)
        self.metrics[name] = {"value": value, "label": label, "verification": True, **extra}


# ------------------------------------------------------------------ comparison helpers (the config's "comparison")
def is_timing(key: str) -> bool:
    last = key.split(".")[-1]
    return last.endswith("wall_s") or last.endswith("_per_s") or last == "wall_total_s" or key.startswith("memory.")


def is_fit_key(key: str) -> bool:
    parts = key.split(".")
    return len(parts) == 3 and parts[0] == "qpred" and parts[2] == "fit"


def canon(entry: dict, value_only: bool = False, fit: bool = False) -> str:
    e = {k: v for k, v in entry.items() if k not in FLAGS}
    if value_only:
        e = {"value": e.get("value")}
    if fit and isinstance(e.get("value"), dict):
        e["value"] = {k: v for k, v in e["value"].items() if k not in FIT_TIMING}
    return json.dumps(hashing.normalize_json(e, allow_nan=True), sort_keys=True)


def compare(ref: dict, new: dict, keys: list[str], value_only: tuple[str, ...] = ()) -> list[dict]:
    bad = []
    for k in keys:
        if k not in new:
            bad.append({"key": k, "missing": True})
            continue
        a = canon(ref[k], value_only=k in value_only, fit=is_fit_key(k))
        b = canon(new[k], value_only=k in value_only, fit=is_fit_key(k))
        if a != b:
            bad.append({"key": k, "reference": a[:300], "new": b[:300]})
    return bad


def names_in(key: str, names: set[str]) -> set[str]:
    return set(key.split(".")) & names


def all_checkpoints(train_run: Path) -> set[str]:
    return set(C.checkpoint_names(train_run))


def npz_equal(a: Path, b: Path) -> bool:
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        if sorted(x.files) != sorted(y.files):
            return False
        return all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes() for k in x.files)


def bytes_equal(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def metrics_of(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text())


def current_metrics(rec: Rec, out: Path) -> dict:
    """This run's metrics as written (the file when run_logged holds the run, else the normalised in-memory dict)."""
    rec.flush()
    f = out / "metrics.json"
    if f.exists():
        return json.loads(f.read_text())
    return {k: hashing.normalize_json(v, allow_nan=True) for k, v in rec.metrics.items()}


# ------------------------------------------------------------------ V2: features
STAGE_LEVEL_PREFIXES = ("docs.", "data.", "prefix.", "control.", "check.docs.", "check.config.")
STAGE_LEVEL_KEYS = ("config_params", "features.train_run")


def stage_features(args, vcfg: dict, ctl: dict, rec: Rec) -> None:
    refs = vcfg["references"]
    tr, ref_f = R.rp(refs["train_run"]), R.rp(refs["features_run"])
    ns = argparse.Namespace(checkpoints=vcfg["checkpoints"]["features"], engine=vcfg["engine"])
    t0 = time.time()
    out = S.features_core(ns, ctl, rec, tr)
    rec.m("verify.features.core_wall_s", time.time() - t0)
    stage_failed = list(rec.failed)
    names = C.resolve_names(vcfg["checkpoints"]["features"], tr)
    new, ref = current_metrics(rec, out), metrics_of(ref_f)
    # F1 files
    f1 = {}
    for n in names:
        for pat in ("features_{}.npz", "fires_{}.npy", "dim_{}.npy"):
            f = pat.format(n)
            f1[f] = (out / f).exists() and (ref_f / f).exists() and hashing.sha256_file(out / f) == hashing.sha256_file(ref_f / f)
        f1[f"features.{n}.file_sha256"] = new.get(f"features.{n}.file_sha256", {}).get("value") == ref[f"features.{n}.file_sha256"]["value"]
    rec.check("verify.F1_files", all(f1.values()), detail=f1)
    # F2 metrics
    others = all_checkpoints(tr) - set(names)
    keys = sorted(k for k in ref if not is_timing(k) and not names_in(k, others)
                  and (k.startswith(STAGE_LEVEL_PREFIXES) or k in STAGE_LEVEL_KEYS or names_in(k, set(names))))
    bad = compare(ref, new, keys)
    rec.check("verify.F2_metrics", not bad, compared=len(keys), mismatches=bad[:40])
    # F3 tables and duplicate
    dp = vcfg["checkpoints"]["dedupe_pair"]
    comp = [n for n in names if n != dp["copy"]]
    f3 = {"features_duplicate": new.get(f"features.{dp['copy']}.duplicate_of", {}).get("value") == dp["source"],
          "prefix_counts_duplicate": new.get(f"prefix_counts.{dp['copy']}.duplicate_of", {}).get("value") == dp["source"],
          "no_table_for_duplicate": not (out / f"prefix_counts_{dp['copy']}.npz").exists()}
    for n in comp:
        f = out / f"prefix_counts_{n}.npz"
        f3[f"table_{n}"] = f.exists() and hashing.sha256_file(f) == new.get(f"prefix_counts.{n}.file_sha256", {}).get("value")
    rec.check("verify.F3_tables", all(f3.values()), detail=f3)
    rec.check("verify.F4_clean", not stage_failed, stage_failed_checks=stage_failed)
    rec.m("verify.features.timing", {n: {"prefix_pass_wall_s_new_one_pass": new.get(f"prefix_fires.{n}.wall_s", {}).get("value"),
                                         "prefix_fires_wall_s_reference": ref.get(f"prefix_fires.{n}.wall_s", {}).get("value"),
                                         "table_save_wall_s_new": new.get(f"prefix_counts.{n}.save_wall_s", {}).get("value")}
                                     for n in comp},
          note="the reference pass computed the fire counts only; Q_pred's counting pass is saved by the one pass (O10)")


# ------------------------------------------------------------------ V3: qpred
Q_NOT_COMPARED = ("qpred.checkpoints", "qpred.summary", "stage", "failed_checks", "qpred.test_accesses_before",
                  "qpred.test_accesses_after", "wall_total_s")


def stage_qpred(args, vcfg: dict, ctl: dict, rec: Rec) -> None:
    refs = vcfg["references"]
    tr, ref_q = R.rp(refs["train_run"]), R.rp(refs["qpred_run"])
    fr = R.rp(args.features_run)
    ns = argparse.Namespace(checkpoints=vcfg["checkpoints"]["qpred"], engine=vcfg["engine"])
    t0 = time.time()
    out = S.qpred_core(ns, ctl, rec, tr, fr, fast_kernel=True, load_counts=True, dedupe=True)
    core_wall = time.time() - t0
    rec.m("verify.qpred.core_wall_s", core_wall)
    stage_failed = list(rec.failed)
    new, ref = current_metrics(rec, out), metrics_of(ref_q)
    ref_names = [n for n in ref["qpred.checkpoints"]["value"]]
    # Q1
    keys = sorted(k for k in ref if not is_timing(k) and k not in Q_NOT_COMPARED)
    bad = compare(ref, new, keys, value_only=("check.qpred.no_test_access",))
    rec.check("verify.Q1_metrics", not bad, compared=len(keys), mismatches=bad[:40])
    # Q2
    s_new, s_ref = new["qpred.summary"]["value"], ref["qpred.summary"]["value"]
    q2 = {n: json.dumps(s_new.get(n), sort_keys=True) == json.dumps(s_ref.get(n), sort_keys=True) for n in ref_names}
    rec.check("verify.Q2_summary", all(q2.values()), detail=q2)
    # Q3
    q3 = {}
    for n in ref_names:
        q3[f"qpred_{n}.npz"] = (out / f"qpred_{n}.npz").exists() and npz_equal(out / f"qpred_{n}.npz", ref_q / f"qpred_{n}.npz")
        q3[f"qpred_{n}_readout.json"] = bytes_equal(out / f"qpred_{n}_readout.json", ref_q / f"qpred_{n}_readout.json")
    rec.check("verify.Q3_files", all(q3.values()), detail=q3)
    # Q4
    used = new.get("qpred.speedups_used", {}).get("value", {})
    q4 = {"speedups_used": bool(used) and all(used.values())}
    for n in ref_names:
        q4[f"{n}.count_source_loaded"] = new.get(f"qpred.{n}.count_source", {}).get("value") == "loaded"
        q4[f"{n}.kernel_T1GpuG"] = "T1GpuG" in str(new.get(f"qpred.{n}.t1_setup_wall_s", {}).get("kernel", ""))
    rec.check("verify.Q4_paths_used", all(q4.values()), detail=q4)
    # Q5
    dp = vcfg["checkpoints"]["dedupe_pair"]
    cp_, sr = dp["copy"], dp["source"]
    q5 = {"duplicate_of": new.get(f"qpred.{cp_}.duplicate_of", {}).get("value") == sr,
          "state_hash_check": new.get(f"check.{cp_}.state_hash_equals_train_run", {}).get("value") is True}
    src_keys = [k for k in new if k.startswith(f"qpred.{sr}.") and not k.endswith("wall_s")]
    miss = []
    for k in src_keys:
        kk = f"qpred.{cp_}." + k[len(f"qpred.{sr}."):]
        e = new.get(kk)
        ok = (e is not None and e.get("label") == "derived" and e.get("copied_from") == sr
              and canon(new[k], value_only=True, fit=is_fit_key(k)) == canon(e, value_only=True, fit=is_fit_key(kk)))
        if not ok:
            miss.append(kk)
    q5["copied_metrics"] = not miss
    q5["files"] = (bytes_equal(out / f"qpred_{cp_}.npz", out / f"qpred_{sr}.npz")
                   and bytes_equal(out / f"qpred_{cp_}_readout.json", out / f"qpred_{sr}_readout.json"))
    rec.check("verify.Q5_copy", all(q5.values()), detail=q5, compared=len(src_keys), mismatched=miss[:20])
    rec.check("verify.Q6_clean", not stage_failed and new.get("check.qpred.no_test_access", {}).get("value") is True,
              stage_failed_checks=stage_failed)
    per = {}
    for n in ref_names:
        fr_new, fr_ref = new[f"qpred.{n}.fit"]["value"], ref[f"qpred.{n}.fit"]["value"]
        per[n] = {"fit_wall_s": [fr_ref["wall_s"], fr_new["wall_s"]], "s_per_evaluation": [fr_ref["s_per_evaluation"], fr_new["s_per_evaluation"]],
                  "function_evaluations": [fr_ref["function_evaluations"], fr_new["function_evaluations"]],
                  "count_pass_wall_s_reference": ref.get(f"qpred.{n}.count_pass_wall_s", {}).get("value"),
                  "count_load_wall_s_new": new.get(f"qpred.{n}.count_load_wall_s", {}).get("value"),
                  "t1_setup_wall_s_new": new.get(f"qpred.{n}.t1_setup_wall_s", {}).get("value")}
    rec.m("verify.qpred.timing", {"reference_wall_total_s": ref["wall_total_s"]["value"], "new_core_wall_s": core_wall,
                                  "per_checkpoint_reference_then_new": per,
                                  "note": "the reference ran 2026-09-25 08:18 UTC with other jobs on the GPU; this run shares the GPU too: "
                                          "the per-evaluation ratio of the crosscheck stage is the like-for-like kernel figure"})


# ------------------------------------------------------------------ V4: crosscheck
def kernel_points(used: np.ndarray, fit_rows: dict, readout: Path, n_buckets: int) -> dict:
    """The config's kernel_points."""
    def vec(v):
        a = np.zeros(n_buckets)
        a[used] = v
        return a
    rd = json.loads(readout.read_text())
    sol = np.zeros(n_buckets)
    for g, a in rd["alpha_by_group"].items():
        sol[int(g)] = float(a)
    a_init = 1.0 / max(1.0, float(np.diff(fit_rows["rp"]).mean())) if fit_rows["n"] else 1.0
    rng = np.random.default_rng(20260925)
    half = sol.copy()
    half[used[1::2]] = 0.0
    return {"init": (vec(np.full(used.shape[0], a_init)), 1.0), "logged_solution": (sol, float(rd["beta0"])),
            "random": (vec(rng.normal(0.05, 0.05, used.shape[0])), 0.8), "half_zero": (half, 0.5),
            "large": (vec(np.full(used.shape[0], 3.0)), -2.0)}


def stage_crosscheck(args, vcfg: dict, ctl: dict, rec: Rec) -> None:
    import cupy as cp

    import cvtm_t1_readout as T1
    from tmlm.data.corpus_store import load_split, split_dir
    from tmlm.data.evalsets import load_evalset, test_access_count

    refs = vcfg["references"]
    tr, ref_q = R.rp(refs["train_run"]), R.rp(refs["qpred_run"])
    fr, v3 = R.rp(args.features_run), R.rp(args.qpred_run)
    p = ctl["params"]
    qp = p["qpred"]
    out = C.out_dir("crosscheck")
    ta0 = test_access_count()
    arm, seed = C.train_run_identity(tr)
    base, _ = C.CONTROL_ARMS[arm]
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    prefix_t = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    V = int(p["vocab"])
    val = P.with_tables(load_split(C.CORPUS, V, "validation"), kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    ev = load_evalset(C.CORPUS, V, "validation", "subword")
    targets = np.asarray(ev["targets"], np.int64)
    hidden = np.asarray(ev["hidden"], np.uint8)
    gold = np.asarray(val.tok)[targets].astype(np.int64)
    vmeta = json.loads((split_dir(C.CORPUS, V, "validation") / "meta.json").read_text())
    R.dataset_hashes({f"{C.CORPUS}_bpe{V}_validation_tok": vmeta["sha256"]["tok.npy"],
                      f"evalset_validation_subword_V{V}": ev["meta"]["sha256_combined"]})
    _, fit_pos, cnt_pos = C.qpred_split(prefix)
    tok = np.asarray(prefix.tok).astype(np.int64)
    zero = np.zeros(int(prefix.tok.shape[0]), np.uint8)
    chunk = int(p["forward_chunk"])
    check_cnt = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(qp["checks"]["cross_engine_count_rows"]), 3)
    index, why_not = S.prefix_count_index(fr, tr, arm, seed)
    rec.m("verify.crosscheck.features_run", str(fr), tables=len(index), unusable_reason=why_not)
    hashes = {}
    for name in vcfg["checkpoints"]["kernel_tables"]:
        cfg, st, meta = C.load_named_checkpoint(tr, name, arm, seed, p, rec)
        hashes[name] = meta["state_hash"]
        Cn = int(cfg.n_clauses)
        # ---- the registered tables and fired lists
        eng_g = R.make_engine("cuda", cfg, prefix_t)
        eng_g.load_state(st)
        t0 = time.time()
        counts, n_ev, got = C.count_table(eng_g, "cuda", cnt_pos, tok[cnt_pos], Cn, V, chunk, zero, check_cnt)
        count_wall = time.time() - t0
        fptr, fids = C.fit_rows_fired(eng_g, "cuda", fit_pos, zero, chunk)
        del eng_g
        cp.get_default_memory_pool().free_all_blocks()
        # ---- K2: the loaded table (through the Q_pred loader) against the recomputed one
        expected = QF.pass_meta(meta["state_hash"], prefix, prefix_t, fit_pos, cnt_pos, check_cnt, cfg, "cuda")
        t0 = time.time()
        loaded, reason = S.load_matching_counts(index, why_not, expected)
        k2 = {"loaded": loaded is not None, "reason": reason, "load_wall_s": time.time() - t0, "registered_count_pass_wall_s": count_wall}
        if loaded is not None:
            dense = loaded.counts_dense()
            k2["counts_equal"] = bool(dense.dtype == counts.dtype and dense.shape == counts.shape and np.array_equal(dense, counts))
            del dense
            k2["events_equal"] = int(loaded.n_events) == int(n_ev)
            lg = loaded.got()
            k2["check_rows_equal"] = sorted(lg) == sorted(got) and all(np.array_equal(lg[r], got[r]) and lg[r].dtype == got[r].dtype for r in got)
            k2["fit_ptr_equal"] = bool(np.array_equal(loaded.fit_ptr, fptr))
            k2["fit_ids_equal"] = bool(loaded.fit_ids.dtype == fids.dtype and np.array_equal(loaded.fit_ids, fids))
        del loaded
        rec.check(f"verify.K2_loaded_table.{name}", all(v for k, v in k2.items() if k.endswith(("equal", "loaded"))) and len(k2) > 4, **k2)
        # ---- K1: the kernels on the registered tables and rows
        n_inc = T1.included_literals(st)
        tab = C.Tables(counts, n_inc, tok[cnt_pos], V)
        del counts
        fit_rows = tab.rows(fptr, fids, tok[fit_pos])
        eng_va = R.make_engine("cpu", cfg, val)
        eng_va.load_state(st)
        vptr, vids = C.api_fired_lists(eng_va, targets, hidden)
        del eng_va
        val_rows = tab.rows(vptr, vids, gold)
        cp.get_default_memory_pool().set_limit(size=int(float(qp["gpu_pool_limit_gib"]) * (1 << 30)))
        t0 = time.time()
        t_reg = T1.T1Gpu(V, tab.indptr, tab.idx, tab.val, tab.lam, tab.inv_n, tab.p_uni)
        setup_reg = time.time() - t0
        t0 = time.time()
        t_fast = QF.T1GpuG(V, tab.indptr, tab.idx, tab.val, tab.lam, tab.inv_n, tab.p_uni)
        setup_fast = time.time() - t0
        g_cpu_diff = float(np.abs(t_fast.gval() - tab.gval).max()) if tab.gval.shape[0] else 0.0
        used = np.unique(fit_rows["bkt"])
        pts = kernel_points(used, fit_rows, ref_q / f"qpred_{name}_readout.json", T1.N_BUCKETS)
        cells, n_bad = {}, 0
        for rname, rows in (("fit", fit_rows), ("validation", val_rows)):
            ch_r = t_reg.prepare(rows["rp"], rows["comp"], rows["bkt"], rows["gold"])
            ch_f = t_fast.prepare(rows["rp"], rows["comp"], rows["bkt"], rows["gold"])
            npairs = int(rows["comp"].shape[0])
            for pname, (alpha, beta0) in pts.items():
                for wg in (True, False):
                    o_r = t_reg.run(ch_r, rows["n"], npairs, alpha, beta0, wg)
                    o_f = t_fast.run(ch_f, rows["n"], npairs, alpha, beta0, wg)
                    eq = QF.outputs_bit_identical(o_r, o_f)
                    cells[f"{rname}.{pname}.grad{int(wg)}"] = eq
                    n_bad += sum(not v for v in eq.values())
                    if not all(eq.values()):
                        rec.log(f"K1 {name} {rname} {pname} grad {wg}: differing outputs {[k for k, v in eq.items() if not v]}")
            del ch_r, ch_f
        rec.check(f"verify.K1_kernel.{name}", n_bad == 0 and len(cells) == 2 * len(pts) * 2, cells=cells, differing_outputs=n_bad,
                  rows={"fit": int(fit_rows["n"]), "validation": int(val_rows["n"])},
                  pairs={"fit": int(fit_rows["comp"].shape[0]), "validation": int(val_rows["comp"].shape[0])},
                  points={k: {"beta0": b, "nonzero_alpha": int(np.count_nonzero(a))} for k, (a, b) in pts.items()})
        # ---- interleaved timing (fit rows, init point, with gradients)
        ch_r = t_reg.prepare(fit_rows["rp"], fit_rows["comp"], fit_rows["bkt"], fit_rows["gold"])
        ch_f = t_fast.prepare(fit_rows["rp"], fit_rows["comp"], fit_rows["bkt"], fit_rows["gold"])
        a0, b0 = pts["init"]
        tr_, tf_ = [], []
        for _ in range(3):
            cp.cuda.Device().synchronize()
            t0 = time.time()
            t_reg.run(ch_r, fit_rows["n"], int(fit_rows["comp"].shape[0]), a0, b0, True)
            tr_.append(time.time() - t0)
            t0 = time.time()
            t_fast.run(ch_f, fit_rows["n"], int(fit_rows["comp"].shape[0]), a0, b0, True)
            tf_.append(time.time() - t0)
        rec.m(f"verify.kernel_timing.{name}", {"registered_s_per_eval": tr_, "fast_s_per_eval": tf_,
                                               "median_registered": float(np.median(tr_)), "median_fast": float(np.median(tf_)),
                                               "ratio_registered_over_fast": float(np.median(tr_) / max(1e-12, np.median(tf_))),
                                               "setup_registered_s": setup_reg, "setup_fast_s": setup_fast,
                                               "pairs_stored": int(tab.idx.shape[0]),
                                               "max_abs_gval_gpu_minus_cpu_reference": g_cpu_diff},
              note="interleaved on a GPU shared with other jobs (shared lock); the gval difference to the CPU k_gval is information, not a rule")
        del t_reg, t_fast, ch_r, ch_f, tab, fit_rows, val_rows
        cp.get_default_memory_pool().free_all_blocks()
        rec.flush()
    # ---- K3: young_b77 by the registered path against V3's copies
    dp = vcfg["checkpoints"]["dedupe_pair"]
    cp_, sr = dp["copy"], dp["source"]
    cfg, st, meta_c = C.load_named_checkpoint(tr, cp_, arm, seed, p, rec)
    if sr not in hashes:
        hashes[sr] = C.load_named_checkpoint(tr, sr, arm, seed, p, rec)[2]["state_hash"]
    t0 = time.time()
    C.qpred_checkpoint(cp_, cfg, st, prefix_t, val, targets, hidden, fit_pos, cnt_pos, qp, "cuda", chunk, rec, out)
    reg_wall = time.time() - t0
    new, v3m = current_metrics(rec, out), metrics_of(v3)
    k3 = {"state_hashes_equal": meta_c["state_hash"] == hashes[sr],
          "hash_checks": new.get(f"check.{cp_}.state_hash_equals_train_run", {}).get("value") is True
          and new.get(f"check.{sr}.state_hash_equals_train_run", {}).get("value") is True}
    pre = f"qpred.{cp_}."
    copied = {k for k, e in v3m.items() if k.startswith(pre) and e.get("copied_from") == sr}
    computed = {k for k in new if k.startswith(pre) and not is_timing(k)}
    missing_in_registered = sorted(k for k in copied - computed if k != pre + "count_source")
    not_copied = sorted(computed - copied)
    diff = [k for k in sorted(copied & computed)
            if canon(v3m[k], value_only=True, fit=is_fit_key(k)) != canon(new[k], value_only=True, fit=is_fit_key(k))]
    k3["copied_keys_compared"] = len(copied & computed)
    k3["values_equal"] = not diff
    k3["registered_keys_all_copied"] = not not_copied
    k3["copied_keys_all_computed"] = not missing_in_registered
    k3["npz_equal"] = npz_equal(out / f"qpred_{cp_}.npz", v3 / f"qpred_{cp_}.npz")
    k3["readout_equal"] = bytes_equal(out / f"qpred_{cp_}_readout.json", v3 / f"qpred_{cp_}_readout.json")
    rec.check("verify.K3_copy_equals_computed", all(v for k, v in k3.items() if isinstance(v, bool)), **k3,
              differing=diff[:20], not_copied=not_copied[:20], missing_in_registered=missing_in_registered[:20])
    fit_y = new.get(f"qpred.{cp_}.fit", {}).get("value", {})
    rec.m("verify.crosscheck.registered_path_timing", {
        "checkpoint": cp_, "per_checkpoint_wall_s": reg_wall,
        "count_pass_wall_s": new.get(f"qpred.{cp_}.count_pass_wall_s", {}).get("value"),
        "fit_wall_s": fit_y.get("wall_s"), "s_per_evaluation": fit_y.get("s_per_evaluation"),
        "function_evaluations": fit_y.get("function_evaluations")},
        note="registered path (count pass, T1Gpu fit, CPU validation scoring) on the state of c0y: the contemporaneous 'before' for V3's c0y")
    ta1 = test_access_count()
    rec.check("verify.no_test_access", ta1 == ta0, before=ta0, after=ta1)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="verification of the exact Q_pred speedups O4, O10, O12 (docs/13 A8 item 7)")
    ap.add_argument("--stage", required=True, choices=["features", "qpred", "crosscheck"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--features-run", default=None)
    ap.add_argument("--qpred-run", default=None)
    args = ap.parse_args(argv)
    vcfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ctl = C.load_config(ROOT / vcfg["controls_config"])
    if args.stage in ("qpred", "crosscheck") and not args.features_run:
        raise SystemExit(f"--stage {args.stage} needs --features-run (the V2 run)")
    if args.stage == "crosscheck" and not args.qpred_run:
        raise SystemExit("--stage crosscheck needs --qpred-run (the V3 run)")
    rec = Rec(False)
    rec.m("stage", args.stage)
    rec.m("config_params", ctl["params"])
    rec.m("verify.config_sha256", hashing.sha256_file(Path(args.config)))
    C.check_shared_params(ctl, rec)
    {"features": stage_features, "qpred": stage_qpred, "crosscheck": stage_crosscheck}[args.stage](args, vcfg, ctl, rec)
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
