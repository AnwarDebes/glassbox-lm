"""Cross-check of the early CVTM-core estimate with the repository combination module
`tmlm.cvtm.mixture` (docs/11 section 11.4). EXPLORATORY: not pre-registered, tunes nothing, tests no
docs/13 threshold. Validation split only; the test split is never loaded. CPU only, one thread.

Why: `experiments/cvtm_early_fulldata.py` (run results/20260924T054225Z_cvtm_early_fulldata_exploratory/)
had to implement the combination itself because `src/tmlm/cvtm/mixture.py` did not exist when it
started. The module appeared during that run and its tests pass (tests/test_cvtm_mixture.py). The two
differ in one point the spec leaves open: how the pooled weights of the Dirichlet prior are formed
(the early script re-estimates them inside the bucket EM as mean responsibilities; the module fits
them first by a separate EM over all rows and gives buckets without fit rows the pooled weights).
This run computes the same rows with the module, so the early estimate is known to carry over to the
estimator the frozen implementation will use.

Inputs reused from the early run (not recomputed): the per-token log2 P_KN of the KN-5 fitted on all
training documents on every validation position (row `kn5` of per_token_log2.npz, equal to the M2
comparator 1.3501314 bit for bit) and k* on validation. P_KN = exp2(log2 P_KN); the round trip error
is logged. Recomputed: the fit documents H (their index hash must equal the early run's), the KN-5
without H (for P_KN and k* on H), the document features (current `tmlm.cvtm.doc_clauses`).

Checks: `mixture.kn_gold_and_depth` equals the early script's `kn_gold` on H (P_KN and k*);
`mixture.gold_components` equals the early script's `components`; the early estimator re-run on
these inputs reproduces the early run's rows; the module's MAP objective and pooled likelihood never
decrease.

Launch (single thread, nice 19):

    nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --limit-gib 30 -- scripts/run_logged.py --no-gpu-sampler \\
        --name cvtm_early_fulldata_mixcheck_exploratory --config experiments/cvtm/configs/cvtm_early_fulldata_mixcheck.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_early_fulldata_mixcheck.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import cvtm_early_fulldata as E  # noqa: E402
from tmlm.cvtm import mixture as MX  # noqa: E402
from tmlm.cvtm.doc_clauses import doc_features  # noqa: E402
from tmlm.data.corpus_store import load_split, raw_dir, split_dir  # noqa: E402
from tmlm.eval.baselines.ngram import KNModel  # noqa: E402

RUN1 = ROOT / "results" / "20260924T054225Z_cvtm_early_fulldata_exploratory"
ROW_COMPONENTS = {
    "kn5_D0": ("S", "D0"),
    "kn5_D0to4": ("S", "D0", "D1", "D2", "D3", "D4"),
    "cvtm_core": ("S", "D0", "D1", "D2", "D3", "D4", "P"),
}
T0 = time.time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}
    if isinstance(value, (int, float, str, bool)) or value is None:
        log(f"{name} = {value}")


def flush() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn-peak-gib", type=float, default=18.0,
                    help="expected peak of the KN-5 fit without H (measured 17.0 GiB peak RSS in the early run)")
    ap.add_argument("--guard-gib", type=float, default=30.0)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing only: 3M-token train prefix, 6 validation docs, P_KN on validation from a KN-5 "
                         "fitted on the prefix instead of the early run's file; not logged")
    args = ap.parse_args()
    if args.smoke and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py")
    M("status", "EXPLORATORY cross-check of results/20260924T054225Z_cvtm_early_fulldata_exploratory with "
      "tmlm.cvtm.mixture; not pre-registered; tunes nothing", label="unverified")
    M("args", vars(args))
    for name in ("mixture.py", "doc_clauses.py"):
        M(f"sha256.src_tmlm_cvtm_{name}", hashlib.sha256((ROOT / "src/tmlm/cvtm" / name).read_bytes()).hexdigest())
    sampler = E.CgroupSampler()
    r1 = json.loads((RUN1 / "metrics.json").read_text())
    npz = np.load(RUN1 / "per_token_log2.npz")

    # ---- data and H
    tr = load_split(E.CORPUS, E.V, "train", check=True)
    va = load_split(E.CORPUS, E.V, "validation", check=True)
    tr_meta = json.loads((split_dir(E.CORPUS, E.V, "train") / "meta.json").read_text())
    va_meta = json.loads((split_dir(E.CORPUS, E.V, "validation") / "meta.json").read_text())
    raw_meta = json.loads((raw_dir(E.CORPUS, "validation") / "meta.json").read_text())
    from tmlm.common import runlog
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({f"train_tok_V{E.V}": tr_meta["sha256"]["tok.npy"],
                                     f"validation_tok_V{E.V}": va_meta["sha256"]["tok.npy"],
                                     "validation_text_utf8": raw_meta["sha256"]["text.utf8"],
                                     "validation_doc_bytes": raw_meta["sha256"]["doc_bytes.npy"]})
    doc_bytes_off = np.load(raw_dir(E.CORPUS, "validation") / "doc_bytes.npy").astype(np.int64)
    n_bytes = int(raw_meta["n_bytes"])
    assert n_bytes == int(doc_bytes_off[-1])
    bytes_doc = np.diff(doc_bytes_off).astype(np.float64)
    tok_tr = np.asarray(tr.tok)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    tok_va = np.ascontiguousarray(np.asarray(va.tok))
    ds_va = np.asarray(va.doc_start).astype(np.int64)
    if args.smoke:
        d = int(np.searchsorted(ds_tr, 3_000_000, side="right")) - 1
        ds_tr = ds_tr[:d + 1]
        tok_tr = tok_tr[:int(ds_tr[-1])]
        ds_va = ds_va[:7]
        tok_va = np.ascontiguousarray(tok_va[:int(ds_va[-1])])
        bytes_doc = bytes_doc[:6]
        n_bytes = int(bytes_doc.sum())
    tok_tr = np.ascontiguousarray(tok_tr)
    n_docs_va = ds_va.shape[0] - 1
    hmask = E.fit_doc_mask(ds_tr.shape[0] - 1)
    h_sha = hashlib.sha256(np.flatnonzero(hmask).astype("<i8").tobytes()).hexdigest()
    if not args.smoke:
        M("H.doc_index_sha256_equals_early_run", h_sha == r1["H.doc_index_sha256"]["value"])
        assert h_sha == r1["H.doc_index_sha256"]["value"]
    tok_h, ds_h = E.select_docs(tok_tr, ds_tr, hmask)
    tok_noh, ds_noh = E.select_docs(tok_tr, ds_tr, ~hmask)

    # ---- document features (current module)
    t0 = time.time()
    feats_h = doc_features(tok_h, ds_h, orders=E.ORDERS, window=None, ptr_min=E.PTR_MIN, ptr_max=E.PTR_MAX)
    feats_va = doc_features(tok_va, ds_va, orders=E.ORDERS, window=None, ptr_min=E.PTR_MIN, ptr_max=E.PTR_MAX)
    M("doc_clauses.batch_wall_s", round(time.time() - t0, 1), unit="s")

    # ---- validation P_KN (all) and k* from the early run
    if args.smoke:
        kn_all = KNModel(E.ORDER, E.V, discount_mode="modified", pad=True).fit(tok_tr, ds_tr)
        p_tmp, k_tmp = E.kn_gold(kn_all, tok_va, ds_va)
        del kn_all
        npz = {"kn5": np.log2(p_tmp), "kstar": k_tmp}
        r1 = {f"{r}.validation.bits_per_byte": {"value": float("nan")} for r in ROW_COMPONENTS}
        r1["kn5_noH.H.bits_per_token"] = {"value": float("nan")}
    lp_kn = npz["kn5"]
    p_va = np.exp2(lp_kn)
    M("p_kn_validation.exp2_roundtrip_max_abs_log2_error", float(np.abs(np.log2(p_va) - lp_kn).max()))
    ks_va = npz["kstar"].astype(np.int64)
    bpb_kn = float(-lp_kn.sum() / n_bytes)
    M("kn5.validation.bits_per_byte", bpb_kn, note="from the early run's per-token log2 (KN-5 on all)")

    # ---- KN-5 without H: P_KN and k* on H
    E.wait_for_memory(args.kn_peak_gib, "kn5_noH fit", args.guard_gib)
    t0 = time.time()
    kn = KNModel(E.ORDER, E.V, discount_mode="modified", pad=True).fit(tok_noh, ds_noh)
    M("kn5_noH.fit_wall_s", round(time.time() - t0, 1), unit="s", note="single thread, nice 19, contended host")
    E.mem("after kn5_noH fit")
    p_h, ks_h = MX.kn_gold_and_depth(kn, tok_h, ds_h)
    p_h2, ks_h2 = E.kn_gold(kn, tok_h, ds_h)
    M("check.kn_gold_and_depth_equals_early_kn_gold", bool(np.array_equal(p_h, p_h2) and np.array_equal(ks_h, ks_h2.astype(np.int64))))
    M("kn5_noH.H.bits_per_token", float(-np.log2(p_h).mean()))
    M("check.kn5_noH.H.bits_per_token_equals_early_run", float(-np.log2(p_h).mean()) == r1["kn5_noH.H.bits_per_token"]["value"])
    del kn, tok_noh, ds_noh
    gc.collect()
    E.mem("after freeing kn5_noH")
    flush()

    # ---- components
    pc_h, act_h = MX.gold_components(p_h, feats_h, tok_h)
    pc_va, act_va = MX.gold_components(p_va, feats_va, tok_va)
    e_pc_h, e_fire_h = E.components(p_h, feats_h, tok_h)
    e_pc_va, e_fire_va = E.components(p_va, feats_va, tok_va)
    M("check.gold_components_equal_early_components",
      bool(np.array_equal(pc_h, e_pc_h) and np.array_equal(act_h, e_fire_h)
           and np.array_equal(pc_va, e_pc_va) and np.array_equal(act_va, e_fire_va)))
    cnt_h, cnt_va = MX.counts_matrix(feats_h), MX.counts_matrix(feats_va)

    # ---- rows with the module; the early estimator re-run on the same inputs
    doc_of_va = np.repeat(np.arange(n_docs_va), np.diff(ds_va))
    idx = np.random.default_rng(np.random.SeedSequence(list(E.BOOT_SEED))).integers(0, n_docs_va, size=(E.N_BOOT, n_docs_va))
    bits_kn_doc = np.bincount(doc_of_va, weights=-lp_kn, minlength=n_docs_va)
    out = {}
    for row, comps in ROW_COMPONENTS.items():
        t0 = time.time()
        b_h, nb = MX.bucket_ids(ks_h, cnt_h, feats_h.ptr_len, comps)
        b_va, nb2 = MX.bucket_ids(ks_va, cnt_va, feats_va.ptr_len, comps)
        assert nb == nb2
        trace: dict = {}
        W = MX.fit_map_em(pc_h, act_h, b_h, E.EM_ITERS, E.PRIOR, n_buckets=nb, components=comps, trace=trace)
        g = MX.gold_prob(pc_va, act_va, b_va, W, comps)
        lg = np.log2(np.maximum(g, np.finfo(np.float64).tiny))
        out[row] = lg
        bpb = float(-lg.sum() / n_bytes)
        early_bpb = r1[f"{row}.validation.bits_per_byte"]["value"]
        M(f"mixture_module.{row}.validation.bits_per_byte", bpb)
        M(f"mixture_module.{row}.validation.bits_per_token", float(-lg.mean()))
        M(f"mixture_module.{row}.minus_early_estimator.bits_per_byte", bpb - early_bpb, label="derived",
          note="module estimator minus the early run's value of the same row")
        obj = np.asarray(trace["bucket_map_objective"])
        pll = np.asarray(trace["pooled_ll"])
        M(f"mixture_module.{row}.objective_non_decreasing",
          {"bucket_map_objective": bool(np.all(np.diff(obj) >= -1e-12)), "pooled_ll": bool(np.all(np.diff(pll) >= -1e-12))})
        M(f"mixture_module.{row}.pooled_weights", dict(zip(MX.COMPONENTS, trace["pooled_weights"])))
        M(f"mixture_module.{row}.bucket_map_objective_nats_per_row", [float(obj[0]), float(obj[1]), float(obj[-1])],
          note="iterations 0, 1 and 50")
        M(f"mixture_module.{row}.validation_rows_in_buckets_without_fit_rows",
          int((np.bincount(b_h, minlength=nb)[b_va] == 0).sum()))
        pt, ci = E.doc_bootstrap(np.bincount(doc_of_va, weights=-lg, minlength=n_docs_va), bits_kn_doc, bytes_doc, idx)
        M(f"mixture_module.{row}.minus_kn5.bits_per_byte", pt, label="derived", ci95=ci,
          note=f"document-block bootstrap, {n_docs_va} documents, {E.N_BOOT} resamples, seed {list(E.BOOT_SEED)}")
        # the early estimator on exactly these inputs
        orders = tuple(int(c[1]) for c in comps if c.startswith("D"))
        use_ptr = "P" in comps
        f_h = E.restrict(e_fire_h, orders, use_ptr)
        f_va = E.restrict(e_fire_va, orders, use_ptr)
        eb_h = E.buckets(ks_h.astype(np.int8), f_h, feats_h, orders, use_ptr)
        eb_va = E.buckets(ks_va.astype(np.int8), f_va, feats_va, orders, use_ptr)
        lam, _ = E.map_em(e_pc_h, f_h, eb_h, E.bucket_fireset(orders, use_ptr), E.EM_ITERS, E.PRIOR)
        e_lg = E.mixture_log2(e_pc_va, lam[eb_va])
        M(f"check.early_estimator_rerun.{row}.abs_diff_vs_early_run", abs(float(-e_lg.sum() / n_bytes) - early_bpb),
          label="derived", note="same estimator, inputs rebuilt here (current doc_clauses, exp2 of the saved log2 P_KN)")
        M(f"{row}.wall_s", round(time.time() - t0, 1), unit="s")
        flush()
    lg = out["cvtm_core"]
    pt, ci = E.doc_bootstrap(np.bincount(doc_of_va, weights=-lg, minlength=n_docs_va), None, bytes_doc, idx,
                             const=E.TRANSFORMER_BPB)
    M("mixture_module.cvtm_core.minus_transformer600s.bits_per_byte", pt, label="derived", ci95=ci,
      note="1.2341 of results/20260922T172902Z_M2_baselines_transformer/ (transformer_L6_W256) as a constant")
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        np.savez_compressed(Path(run_dir) / "per_token_log2_mixture_module.npz", **out)
    r, pk = E.own_rss_gib()
    M("peak_rss_gib", round(pk, 2), note="VmHWM of this process")
    M("peak_cgroup_working_set_gib", round(sampler.stop() / E.GIB, 2), note="all processes of the container, sampled every 2 s")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
