"""H8.11: the masked CVTM (docs/11 section 11.8; docs/13 amendment A3, H8.11). Library:
`src/tmlm/cvtm/masked.py` (its docstring defines the model). Config (pre-registered settings):
`experiments/cvtm/configs/H8_11_masked.json`. VALIDATION ONLY: the test split is never loaded (the
test-access counter is read before and after and must not change).

Steps:
1. Data: the V 8000 train split (arrays re-hashed), the validation split and its frozen subword eval
   set (42,213 targets, the frozen hidden mask; the set's combined sha256 must equal the one the M2
   comparators logged). Fit documents H = residue 0 (docs/12 F20 b; 304 documents, 1,397,702 tokens,
   asserted by `protocol.check_h`).
2. Per pattern (the ten of `masked.PATTERNS`): exact counts over EVERY training position on the GPU
   (sort based), answered for the fit rows (N and the gold count) and for the validation contexts
   (full successor lists); the same counts on H alone (host); fit-row counts WITHOUT H = all minus H.
   Document-clause candidates of the fit rows (every H position, only its target hidden) and of the
   validation targets (the frozen mask). Pipeline checks: CuPy and numpy counts equal on a train
   prefix; the stream keys equal the positional keys on a sample of H; after the loop, for two
   patterns, counting directly on the documents without H equals the subtraction.
3. EM of the mixture weights on the fit rows (plain minorise-maximise, docs/12 F20 c), trace
   non-decreasing; unfitted buckets resolved by the fixed rule; the S-only ablation fitted the same
   way (diagnostic row, no threshold).
4. Validation scoring with the full distribution over V: top-1 (ties to the lowest id), top-5, bits
   per token; every row sums to one within 1e-9; the gold entry equals the sparse formula (1e-12).
5. Comparators: the three logged numbers are read from their run directories and checked against
   the values cited in docs/13 A3; the left bigram (true context) and KN-5 both sides are recomputed
   per target and must reproduce the logged numbers exactly; they feed the paired document-block
   bootstrap (60 validation documents, 2,000 resamples, labelled derived). CBOW-kNN enters as its
   logged number (its model is not stored).
6. Verdict H8.11 by the registered rule; per-target arrays and weights saved in the run directory.

Launch (shared GPU lock, one CPU slot, mem_guard at its default limit):

    scripts/gpu_run.sh shared -- scripts/cpu_slot.sh 1 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_H8_11_masked --config experiments/cvtm/configs/H8_11_masked.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_masked.py --config experiments/cvtm/configs/H8_11_masked.json

Smoke (not logged, labels "unverified", output under $TMLM_SCRATCH): `--smoke` counts on a train
prefix without a few held-out TRAINING documents, which serve as a pseudo-evaluation set under a
synthetic 15 percent mask. No validation number is computed in smoke mode (the validation split and
eval set are only loaded and hash-checked).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402
from tmlm.cvtm import masked as MK  # noqa: E402
from tmlm.cvtm import protocol  # noqa: E402

GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[cvtm_masked {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if SMOKE:
        label = "unverified"
    METRICS[name] = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_metrics({name: METRICS[name]}, on_conflict="replace")
    if isinstance(value, (int, float, str, bool)) or value is None:
        log(f"{name} = {value}")


def check(name: str, ok: bool, *, reference_check: bool = False, **details) -> None:
    """A pipeline check; a failure is recorded and makes the run exit 1. In smoke mode a
    reference_check (a comparison with a logged full-data number) is recorded but cannot fail."""
    M(f"check.{name}", bool(ok), details=details)
    if not ok and not (SMOKE and reference_check):
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {details}")


def out_dir() -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_masked_unlogged"
    p.mkdir(parents=True, exist_ok=True)
    return p


def rss_peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# ------------------------------------------------------------------ memory rule
CG = Path("/sys/fs/cgroup")


def cgroup() -> tuple[int, int, int]:
    """(memory.current, working set = current - inactive_file, memory.max)."""
    cur = int((CG / "memory.current").read_text().strip())
    mx_txt = (CG / "memory.max").read_text().strip()
    mx = int(mx_txt) if mx_txt != "max" else 1 << 62
    inact = 0
    for line in (CG / "memory.stat").read_text().splitlines():
        k, _, v = line.partition(" ")
        if k == "inactive_file":
            inact = int(v)
            break
    return cur, max(0, cur - inact), mx


def wait_for_memory(peak_gib: float, tag: str, poll_s: int = 120, max_wait_s: int = 6 * 3600) -> None:
    """Start a step expected above 8 GiB only when memory.current + its peak < 0.80 memory.max
    (poll every 2 minutes)."""
    waited = 0
    while True:
        try:
            cur, ws, mx = cgroup()
        except OSError:
            log(f"memory before {tag}: cgroup counters unreadable, proceeding")
            return
        if cur + peak_gib * GIB < 0.80 * mx:
            log(f"memory before {tag}: current {cur / GIB:.1f} GiB (working set {ws / GIB:.1f}) + peak {peak_gib:.0f} GiB "
                f"< 0.80 x {mx / GIB:.0f} GiB (waited {waited} s)")
            M(f"memory.before.{tag}", {"current_gib": cur / GIB, "working_set_gib": ws / GIB, "max_gib": mx / GIB,
                                       "expected_peak_gib": peak_gib, "waited_s": waited})
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory rule: waited {waited} s before {tag}")
        log(f"memory before {tag}: current {cur / GIB:.1f} GiB + peak {peak_gib:.0f} GiB >= 0.80 x {mx / GIB:.0f} GiB; waiting")
        time.sleep(poll_s)
        waited += poll_s


# ------------------------------------------------------------------ data
@dataclass
class Data:
    vocab: int
    tok: np.ndarray            # counting stream: every training document (uint16)
    doc_start: np.ndarray      # int64
    h: protocol.DocSubset      # fit documents H (whole documents of the counting stream)
    h_pos: np.ndarray          # int64: positions of H in the counting stream
    rest: protocol.DocSubset   # the counting documents without H
    ev_tok: np.ndarray         # evaluation stream
    ev_doc_start: np.ndarray
    targets: np.ndarray        # int64, ascending
    hidden: np.ndarray         # uint8 [n_ev]
    label: str


def positions_of_docs(doc_start: np.ndarray, keep_doc: np.ndarray) -> np.ndarray:
    ds = np.asarray(doc_start).astype(np.int64)
    return np.flatnonzero(np.repeat(np.asarray(keep_doc, bool), np.diff(ds))).astype(np.int64)


def make_data(tok: np.ndarray, ds: np.ndarray, doc_index: np.ndarray | None, ev_tok, ev_ds, targets, hidden,
              vocab: int, label: str) -> Data:
    ids = np.arange(ds.shape[0] - 1, dtype=np.int64) if doc_index is None else np.asarray(doc_index, np.int64)
    keep_h = np.isin(protocol.residues(ids), np.asarray(protocol.H_RESIDUES))
    h = protocol.select_docs(tok, ds, keep_h, ids)
    rest = protocol.select_docs(tok, ds, ~keep_h, ids)
    return Data(vocab=vocab, tok=np.ascontiguousarray(tok), doc_start=np.asarray(ds, np.int64), h=h,
                h_pos=positions_of_docs(ds, keep_h), rest=rest, ev_tok=np.ascontiguousarray(ev_tok),
                ev_doc_start=np.asarray(ev_ds, np.int64), targets=np.asarray(targets, np.int64),
                hidden=np.asarray(hidden, np.uint8), label=label)


# ------------------------------------------------------------------ features
def build_features(data: Data, cfg: dict, backend: str) -> dict:
    """Per pattern: fit-row counts without H, document candidates of the fit rows and of the
    evaluation targets, and the successor lists of the evaluation contexts."""
    V = data.vocab
    nH = data.h.n_tokens
    nT = int(data.targets.shape[0])
    gold_h = data.h.tok.astype(np.int64)
    doc_h = data.h.doc_of()
    vis_ev = (1 - data.hidden).astype(np.uint8)
    doc_ev = MK.doc_index_of(data.ev_doc_start, data.ev_tok.shape[0])
    ev_gold = data.ev_tok[data.targets].astype(np.int64)
    s_n_fit = np.zeros((nH, MK.N_PATTERNS), np.int64)
    s_cnt_fit = np.zeros((nH, MK.N_PATTERNS), np.int64)
    d_tok_fit = np.full((nH, MK.N_PATTERNS), -1, np.int64)
    s_n_ev = np.zeros((nT, MK.N_PATTERNS), np.int64)
    s_ref_ev = np.full((nT, MK.N_PATTERNS), -1, np.int64)
    d_tok_ev = np.full((nT, MK.N_PATTERNS), -1, np.int64)
    parts: list[dict] = []
    keys_h_keep: dict[str, np.ndarray] = {}
    checks_cfg = cfg["checks"]
    t0 = time.time()
    st = MK.upload_stream(data.tok, data.doc_start, backend)
    st_h = MK.upload_stream(data.h.tok, data.h.doc_start, "numpy")
    M("features.upload_s", time.time() - t0)
    # prefix streams for the cross-backend check (only when the main backend is not numpy)
    pre = None
    if backend != "numpy":
        ptok, pds = protocol.prefix_documents(data.tok, data.doc_start, int(checks_cfg["prefix_tokens"]))
        pre = (MK.upload_stream(ptok, pds, backend), MK.upload_stream(ptok, pds, "numpy"), ptok, pds)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1108, 2]))
    ks = min(int(checks_cfg["key_sample"]), nH)
    key_sample = np.sort(rng.choice(nH, size=ks, replace=False)) if ks else np.zeros(0, np.int64)
    n_lists = 0
    for pi, (name, offs) in enumerate(MK.PATTERNS):
        tp = time.time()
        kh, _ = MK.pattern_keys(data.h.tok, data.h.doc_start, np.arange(nH), offs, V)
        # the H keys equal the keys of the same positions in the counting stream (whole documents)
        kfull, _ = MK.pattern_keys(data.tok, data.doc_start, data.h_pos[key_sample], offs, V)
        check(f"keys_h_equal_counting_stream.{name}", bool(np.array_equal(kfull, kh[key_sample])), sample=int(ks))
        kv, vv = MK.pattern_keys(data.ev_tok, data.ev_doc_start, np.arange(data.ev_tok.shape[0]), offs, V, vis=vis_ev)
        kt, vt = kv[data.targets], vv[data.targets]
        lk = np.unique(kt[vt])
        res = MK.count_queried(st, offs, V, gold_keys=kh, gold_tok=data.h.tok, list_keys=lk)
        rh = MK.count_queried(st_h, offs, V, gold_keys=kh, gold_tok=data.h.tok)
        n_no = res["gold_n"] - rh["gold_n"]
        c_no = res["gold_cnt"] - rh["gold_cnt"]
        check(f"h_counts_positive.{name}", bool(np.all(rh["gold_n"] >= 1) and np.all(rh["gold_cnt"] >= 1)))
        check(f"noh_counts_valid.{name}", bool(np.all(n_no >= 0) and np.all(c_no >= 0) and np.all(c_no <= n_no)))
        s_n_fit[:, pi] = n_no
        s_cnt_fit[:, pi] = c_no
        if name in checks_cfg["direct_noH_patterns"]:
            keys_h_keep[name] = kh
        cand = MK.doc_candidates_fit(kh, doc_h, MK.right_extent(offs))
        d_tok_fit[:, pi] = np.where(cand >= 0, data.h.tok[np.maximum(cand, 0)].astype(np.int64), -1)
        # evaluation rows
        j = np.searchsorted(lk, kt)
        jc = np.minimum(j, max(0, lk.shape[0] - 1))
        found = vt & (lk.shape[0] > 0) & (lk[jc] == kt) if lk.shape[0] else np.zeros(nT, bool)
        if lk.shape[0]:
            nfound = np.where(found, res["list_n"][jc], 0)
        else:
            nfound = np.zeros(nT, np.int64)
        s_n_ev[:, pi] = nfound
        s_ref_ev[:, pi] = np.where(nfound > 0, n_lists + jc, -1)
        parts.append(res)
        n_lists += lk.shape[0]
        cand_ev = MK.doc_candidates_eval(kv, vv, vis_ev, doc_ev)[data.targets]
        d_tok_ev[:, pi] = np.where(cand_ev >= 0, data.ev_tok[np.maximum(cand_ev, 0)].astype(np.int64), -1)
        # cross-backend check on the prefix
        if pre is not None:
            pst_b, pst_n, ptok, pds = pre
            pk, _ = MK.pattern_keys(ptok, pds, np.arange(ptok.shape[0]), offs, V)
            sel = rng.choice(ptok.shape[0], size=min(50000, ptok.shape[0]), replace=False)
            q_g = pk[sel]
            q_t = rng.integers(0, V, size=sel.shape[0])
            q_t[: sel.shape[0] // 2] = ptok[sel[: sel.shape[0] // 2]]
            q_l = np.unique(np.concatenate([pk[sel[:5000]], rng.integers(0, 2 ** 40, size=200).astype(np.uint64)]))
            a = MK.count_queried(pst_b, offs, V, gold_keys=q_g, gold_tok=q_t, list_keys=q_l)
            b = MK.count_queried(pst_n, offs, V, gold_keys=q_g, gold_tok=q_t, list_keys=q_l)
            same = all(np.array_equal(a[k], b[k]) for k in ("gold_n", "gold_cnt", "list_found", "list_n", "list_ptr",
                                                            "list_tgt", "list_cnt", "n_ctx", "n_pairs", "max_n"))
            check(f"backend_equal_on_prefix.{name}", bool(same), prefix_tokens=int(ptok.shape[0]))
        # statistics
        hits_fit = (d_tok_fit[:, pi] == gold_h)
        fire_fit = d_tok_fit[:, pi] >= 0
        fire_ev = d_tok_ev[:, pi] >= 0
        stats = {
            "table_contexts": res["n_ctx"], "table_pairs": res["n_pairs"], "table_max_count": res["max_n"],
            "count_seconds": res["seconds"], "gpu_pool_bytes_before_release": res.get("pool_bytes_before_release"),
            "fit_rows_s_fires": float((n_no > 0).mean()), "fit_rows_d_fires": float(fire_fit.mean()),
            "fit_rows_d_hit_rate_when_firing": float(hits_fit[fire_fit].mean()) if fire_fit.any() else None,
            "eval_targets_pattern_valid": float(vt.mean()), "eval_targets_s_fires": float((nfound > 0).mean()),
            "eval_targets_d_fires": float(fire_ev.mean()),
            "eval_contexts_queried": int(lk.shape[0]), "eval_list_entries": int(res["list_ptr"][-1]),
        }
        M(f"pattern.{name}", stats, note="table_* = the full count table over every counting position (distinct contexts = S clauses, distinct (context, class) pairs); fit rows = every position of H with counts without H; eval = the evaluation targets under the frozen mask")
        log(f"pattern {name}: {res['n_ctx']} contexts, {res['n_pairs']} pairs, count {res['seconds']:.1f} s, "
            f"fit S {stats['fit_rows_s_fires']:.3f} D {stats['fit_rows_d_fires']:.3f}, eval valid {stats['eval_targets_pattern_valid']:.3f} "
            f"S {stats['eval_targets_s_fires']:.3f} D {stats['eval_targets_d_fires']:.3f} ({time.time() - tp:.1f} s)")
        del kh, kv, vv, kt, vt, cand, cand_ev, rh
        gc.collect()
    st.close()
    if pre is not None:
        pre[0].close()
    del st, pre
    gc.collect()
    # direct counting without H for the named patterns (the subtraction identity on real data)
    if checks_cfg["direct_noH_patterns"]:
        st_r = MK.upload_stream(data.rest.tok, data.rest.doc_start, backend)
        for name in checks_cfg["direct_noH_patterns"]:
            pi = MK.PATTERN_NAMES.index(name)
            r = MK.count_queried(st_r, MK.PATTERNS[pi][1], V, gold_keys=keys_h_keep[name], gold_tok=data.h.tok)
            ok = bool(np.array_equal(r["gold_n"], s_n_fit[:, pi]) and np.array_equal(r["gold_cnt"], s_cnt_fit[:, pi]))
            check(f"noh_subtraction_equals_direct_count.{name}", ok, rows=int(nH))
        st_r.close()
        del st_r
    lists, _bases = MK.concat_lists(parts)
    del parts
    lists.check()
    M("features.eval_list_entries_total", int(lists.tgt.shape[0]))
    M("features.table_total_contexts", int(sum(METRICS[f"pattern.{n}"]["value"]["table_contexts"] for n in MK.PATTERN_NAMES)),
      note="S clauses of the full model = distinct pattern contexts over the counting positions, all ten patterns")
    M("features.table_total_pairs", int(sum(METRICS[f"pattern.{n}"]["value"]["table_pairs"] for n in MK.PATTERN_NAMES)),
      note="exact (clause, class) counts of the full model, all ten patterns")
    M("features.wall_s", time.time() - t0)
    return {"s_n_fit": s_n_fit, "s_cnt_fit": s_cnt_fit, "d_tok_fit": d_tok_fit, "gold_fit": gold_h,
            "s_n_ev": s_n_ev, "s_ref_ev": s_ref_ev, "d_tok_ev": d_tok_ev, "gold_ev": ev_gold, "lists": lists}


# ------------------------------------------------------------------ model rows
def fit_and_score(tag: str, feats: dict, p_uni_fit: np.ndarray, p_uni_all: np.ndarray, cfg: dict, use_d: bool) -> dict:
    em = cfg["em"]
    fr = MK.fit_rows(feats["gold_fit"], p_uni_fit, feats["s_n_fit"], feats["s_cnt_fit"], feats["d_tok_fit"], use_d=use_d)
    fit = MK.fit_em(fr, int(em["max_iter"]), float(em["tol_mean_nats_per_row"]))
    tr = fit["trace"]
    check(f"{tag}.em_trace_non_decreasing", bool(fit["min_step_gain"] >= -1e-12), min_step_gain=fit["min_step_gain"])
    M(f"{tag}.em", {"iterations": fit["iterations"], "stopped_by_tol": fit["stopped_by_tol"],
                    "fit_mean_nats_per_row_first": float(tr[0]), "fit_mean_nats_per_row_last": float(tr[-1]),
                    "last_step_gain": fit["last_step_gain"], "seconds": fit["seconds"], "fit_rows": fr.n},
      note="plain minorise-maximise EM on the fit rows (docs/12 F20 c), from all weights 1; trace = mean log-likelihood per fit row")
    log(f"{tag}: EM {fit['iterations']} iterations (stopped by tol: {fit['stopped_by_tol']}), fit nats per row "
        f"{tr[0]:.6f} -> {tr[-1]:.9f}, last gain {fit['last_step_gain']:.3e}, {fit['seconds']:.1f} s")
    w, rep = MK.resolve_weights(fit["weights"], fit["rows_per_param"])
    er = MK.eval_rows(feats["s_n_ev"], feats["s_ref_ev"], feats["d_tok_ev"], use_d=use_d)
    used_sub = 0
    fitted = fit["rows_per_param"] > 0
    ev_par = er.par[:, :MK.N_PATTERNS]
    used_sub = int(((ev_par >= 0) & ~fitted[np.maximum(ev_par, 0)]).sum())
    M(f"{tag}.weights", weights_report(w, fit["rows_per_param"]), note="weights scaled so that the largest is 1 (the mixture is scale free); a0 = unigram; alpha per (pattern, floor(log2 N)) with its fit rows; beta = document clause confidence per pattern")
    M(f"{tag}.weights_sha256", MK.weights_hash(w))
    M(f"{tag}.unfitted_buckets", {"n_substituted": rep["n_substituted"], "eval_components_on_substituted_buckets": used_sub,
                                  "unfitted_document_clauses": rep["unfitted_d"]},
      note="rule fixed in the config: an unfitted (pattern, level) bucket takes the nearest fitted level of the same pattern (lower on a tie)")
    sc = MK.score(er, w, feats["lists"], p_uni_all, feats["gold_ev"])
    check(f"{tag}.rows_sum_to_one", bool(sc["max_row_sum_error"] < 1e-9), max_row_sum_error=sc["max_row_sum_error"])
    check(f"{tag}.gold_dense_equals_sparse", bool(sc["max_rel_gold_diff"] < 1e-12), max_rel_diff=sc["max_rel_gold_diff"])
    check(f"{tag}.gold_positive", bool(np.all(sc["p_gold"] > 0)))
    summ = MK.topk_summary(sc["rank"], sc["p_gold"])
    summ["scoring_seconds"] = sc["seconds"]
    log(f"{tag}: top-1 {summ['top1']:.6f} top-5 {summ['top5']:.6f} bits per token {summ['bits_per_token']:.4f} ({summ['n']} targets)")
    return {"fit": fit, "weights": w, "resolve": rep, "score": sc, "summary": summ, "eval_rows": er}


def weights_report(w: np.ndarray, rows_per_param: np.ndarray) -> dict:
    out: dict = {"alpha": {}, "beta": {}, "a0": float(w[MK.U_PARAM])}
    for pi, name in enumerate(MK.PATTERN_NAMES):
        lv = {}
        for level in range(MK.LOGN_LEVELS):
            b = pi * MK.LOGN_LEVELS + level
            if rows_per_param[b] > 0:
                lv[str(level)] = {"w": float(w[b]), "fit_rows": int(rows_per_param[b])}
        out["alpha"][name] = lv
        bw = float(w[MK.D_PARAM0 + pi])
        out["beta"][name] = {"w": bw if math.isfinite(bw) else None, "fit_rows": int(rows_per_param[MK.D_PARAM0 + pi])}
    return out


# ------------------------------------------------------------------ comparators
def comparator_rows(data: Data, cfg: dict, backend: str) -> dict:
    """Per-target predictions of the left bigram (true left token, the M2 function) and of KN-5 both
    sides (forward and backward modified KN-5 on every training document, product of experts, the
    visible context), on the evaluation targets."""
    from tmlm.cvtm import kn_gpu
    from tmlm.eval.baselines.ngram import BidirectionalKN, bigram_table, left_bigram_top1, reverse_docs

    V = data.vocab
    out: dict = {}
    t0 = time.time()
    big = bigram_table(data.tok, V)
    lb = left_bigram_top1(data.tok, V, data.targets, data.ev_tok, big)
    ok = data.targets >= 1
    pred = np.full(data.targets.shape[0], -1, np.int64)
    pred[ok] = big[data.ev_tok[data.targets[ok] - 1].astype(np.int64)].argmax(axis=1)
    del big
    gold = data.ev_tok[data.targets].astype(np.int64)
    out["left_bigram"] = {"pred": pred, "top1_function": float(lb), "top1_rows": float((pred == gold).mean()),
                          "seconds": time.time() - t0}
    gc.collect()
    # the memory rule applies to steps expected above 8 GiB: the KN peak scales with the tokens
    peak = float(cfg["compute"]["kn_host_peak_gib"]) * data.tok.shape[0] / 135_246_414
    if peak > 8.0:
        wait_for_memory(peak, "kn5_both_fit")
    t0 = time.time()
    lim = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    fwd = kn_gpu.fit_kn(data.tok, data.doc_start, 5, V, backend=backend, max_device_bytes=lim if backend == "cupy" else None)
    bwd = kn_gpu.fit_kn(reverse_docs(data.tok, data.doc_start), data.doc_start, 5, V, backend=backend,
                        max_device_bytes=lim if backend == "cupy" else None)
    fit_s = time.time() - t0
    bi = BidirectionalKN(fwd, bwd)
    t1 = time.time()
    pk = bi.predict(data.ev_tok, data.ev_doc_start, data.targets, data.hidden, order=5, side="both")
    out["kn5_both"] = {"pred": pk.astype(np.int64), "top1_rows": float((pk == gold).mean()), "fit_seconds": fit_s,
                       "predict_seconds": time.time() - t1,
                       "fit_info": {"fwd": {k: v for k, v in fwd.fit_info.items() if k != "phases_s"},
                                    "bwd": {k: v for k, v in bwd.fit_info.items() if k != "phases_s"}}}
    del fwd, bwd, bi
    gc.collect()
    return out


def logged_comparators(cfg: dict) -> dict:
    out = {}
    for name, c in cfg["comparators"]["rows"].items():
        m = json.loads((ROOT / c["source_run"] / "metrics.json").read_text())
        v = float(m[c["metric_key"]]["value"])
        out[name] = v
        check(f"comparator_logged_matches_cited.{name}", abs(v - float(c["value_cited"])) < 1e-4,
              logged=v, cited=float(c["value_cited"]), source=c["source_run"], key=c["metric_key"])
        M(f"comparator.{name}.top1_logged", v, note=f"from {c['source_run']}metrics.json key {c['metric_key']}")
    return out


# ------------------------------------------------------------------ bootstrap
def doc_bootstrap(hits_by_doc: np.ndarray, n_by_doc: np.ndarray, idx: np.ndarray, const: float = 0.0) -> dict:
    """Ratio of sums sum(hits) / sum(n) over resampled documents (minus const); 95 percent
    percentile interval. Labelled derived."""
    h = np.asarray(hits_by_doc, np.float64)
    n = np.asarray(n_by_doc, np.float64)
    stats = h[idx].sum(axis=1) / n[idx].sum(axis=1) - const
    return {"point": float(h.sum() / n.sum()) - const, "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "n_boot": int(idx.shape[0]), "n_docs": int(idx.shape[1])}


# ------------------------------------------------------------------ main pipeline
def run_pipeline(data: Data, cfg: dict, backend: str, with_comparators: bool = True) -> dict:
    V = data.vocab
    M("data.label", data.label)
    M("data.counting_tokens", int(data.tok.shape[0]))
    M("data.counting_docs", int(data.doc_start.shape[0] - 1))
    M("data.fit_docs", int(data.h.n_docs))
    M("data.fit_rows", int(data.h.n_tokens))
    M("data.eval_tokens", int(data.ev_tok.shape[0]))
    M("data.eval_targets", int(data.targets.shape[0]))
    check("targets_are_hidden", bool(np.all(data.hidden[data.targets] == 1)))
    check("targets_ascending", bool(np.all(np.diff(data.targets) > 0)))
    p_uni_all = MK.add_one_unigram(data.tok, V)
    p_uni_fit = MK.add_one_unigram(data.tok, V, minus=data.h.tok)
    check("unigram_without_h_equals_direct", bool(np.array_equal(p_uni_fit, MK.add_one_unigram(data.rest.tok, V))))
    feats = build_features(data, cfg, backend)
    res = {"feats": feats}
    t0 = time.time()
    main = fit_and_score("cvtm_masked", feats, p_uni_fit, p_uni_all, cfg, use_d=True)
    M("cvtm_masked.validation" if not SMOKE else "cvtm_masked.pseudo_eval", main["summary"],
      note="the masked CVTM: S and D clauses of the ten patterns, lin mixture fitted on H; top-1 argmax over V (ties to the lowest id)")
    abl = fit_and_score("s_only", feats, p_uni_fit, p_uni_all, cfg, use_d=False)
    M("s_only.validation" if not SMOKE else "s_only.pseudo_eval", abl["summary"],
      note="diagnostic ablation (no threshold): the same pipeline without the document clauses")
    M("model.wall_s", time.time() - t0)
    res["main"], res["ablation"] = main, abl
    if with_comparators:
        res["comparators"] = comparator_rows(data, cfg, backend)
    return res


def load_full(cfg: dict) -> Data:
    from tmlm.data.evalsets import load_evalset

    V = int(cfg["data"]["vocab"])
    corpus = cfg["data"]["corpus"]
    t0 = time.time()
    tok, ds = protocol.load_train(corpus, V, check=True)
    tok = np.array(tok, dtype=np.uint16)          # into memory (270 MB), not the memory map
    val = protocol.load_eval_split("validation", corpus, V, check=True)
    sub = load_evalset(corpus, V, "validation", "subword")
    hashes = protocol.write_dataset_hashes(("train", "validation"), corpus, V)
    ev_hash = sub["meta"]["sha256_combined"]
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({f"evalset_validation_subword_V{V}": ev_hash}, on_conflict="replace")
    M("data.dataset_hashes", {**hashes, f"evalset_validation_subword_V{V}": ev_hash})
    check("evalset_is_the_comparators_set", ev_hash == cfg["data"]["evalset_sha256_combined"], got=ev_hash)
    targets = sub["targets"].astype(np.int64)
    check("evalset_target_count", int(targets.shape[0]) == int(cfg["data"]["n_targets"]), got=int(targets.shape[0]))
    data = make_data(tok, ds, None, val.tok, val.doc_start, targets, sub["hidden"].astype(np.uint8), V, "validation")
    protocol.check_h(data.h.n_docs, data.h.n_tokens, data.h.doc_index)
    check("fit_documents_h", True, n_docs=data.h.n_docs, n_tokens=data.h.n_tokens,
          doc_index_sha256=protocol.index_sha256(data.h.doc_index))
    M("data.load_s", time.time() - t0)
    return data


def load_smoke(cfg: dict) -> Data:
    """Train prefix; pseudo-evaluation documents (residues of the config, never H) removed from the
    counting stream, masked at 15 percent by a fixed seed. The validation split is loaded and its
    hashes checked but nothing is scored on it."""
    from tmlm.data.evalsets import load_evalset

    sm = cfg["smoke"]
    V = int(cfg["data"]["vocab"])
    corpus = cfg["data"]["corpus"]
    tok, ds = protocol.load_train(corpus, V, check=False)
    tok_p, ds_p = protocol.prefix_documents(tok, ds, int(sm["train_tokens"]))
    tok_p = np.array(tok_p, dtype=np.uint16)
    ids = np.arange(ds_p.shape[0] - 1, dtype=np.int64)
    ev_docs = np.isin(protocol.residues(ids), np.asarray(sm["eval_residues"]))
    if np.any(np.isin(np.asarray(sm["eval_residues"]), np.asarray(protocol.H_RESIDUES))):
        raise ValueError("smoke evaluation residues must not overlap H")
    cnt = protocol.select_docs(tok_p, ds_p, ~ev_docs, ids)
    ev = protocol.select_docs(tok_p, ds_p, ev_docs, ids)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1108, 1]))
    hidden = (rng.random(ev.n_tokens) < 0.15).astype(np.uint8)
    targets = np.flatnonzero(hidden).astype(np.int64)
    sub = load_evalset(corpus, V, "validation", "subword")
    check("evalset_is_the_comparators_set", sub["meta"]["sha256_combined"] == cfg["data"]["evalset_sha256_combined"])
    check("evalset_target_count", int(sub["targets"].shape[0]) == int(cfg["data"]["n_targets"]))
    return make_data(cnt.tok, cnt.doc_start, cnt.doc_index, ev.tok, ev.doc_start, targets, hidden, V,
                     f"smoke: {int(sm['train_tokens'])}-token prefix, pseudo-eval residues {sm['eval_residues']}")


def verdict(top1: float, logged: dict) -> dict:
    lb, cb, kn = logged["left_bigram_truectx"], logged["cbow_knn_visible"], logged["kn5_both_visible"]
    met = bool(top1 > lb and top1 > cb)
    return {"verdict": "MET" if met else "NOT MET", "top1": top1, "left_bigram_truectx": lb, "cbow_knn_visible": cb,
            "above_left_bigram": bool(top1 > lb), "above_cbow_knn": bool(top1 > cb),
            "kn5_both_visible": kn, "above_kn5_both": bool(top1 > kn), "minus_kn5_both": top1 - kn}


def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "cvtm" / "configs" / "H8_11_masked.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backend", default=None, help="override the config backend (smoke only)")
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    cfg = json.loads(Path(args.config).read_text())
    backend = cfg["compute"]["backend"] if args.backend is None else args.backend
    if args.backend is not None and not SMOKE:
        raise SystemExit("--backend is a smoke option; the logged run uses the config backend")
    from tmlm.data.evalsets import test_access_count

    ta0 = test_access_count()
    M("mode_smoke", SMOKE)
    M("test_accesses_before", ta0)
    if backend == "cupy":
        import cupy
        cupy.get_default_memory_pool().set_limit(size=int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB))
    M("compute.backend", backend)
    data = load_smoke(cfg) if SMOKE else load_full(cfg)
    res = run_pipeline(data, cfg, backend, with_comparators=True)
    main_sc = res["main"]["score"]
    gold = res["feats"]["gold_ev"]
    comp = res["comparators"]
    lb = comp["left_bigram"]
    kn = comp["kn5_both"]
    logged = logged_comparators(cfg)
    check("left_bigram_function_equals_rows", lb["top1_function"] == lb["top1_rows"],
          function=lb["top1_function"], rows=lb["top1_rows"])
    check("left_bigram_reproduces_logged", lb["top1_function"] == logged["left_bigram_truectx"], reference_check=True,
          got=lb["top1_function"], logged=logged["left_bigram_truectx"])
    check("kn5_both_reproduces_logged", kn["top1_rows"] == logged["kn5_both_visible"], reference_check=True,
          got=kn["top1_rows"], logged=logged["kn5_both_visible"])
    M("comparator.left_bigram_truectx.top1_recomputed", lb["top1_rows"])
    M("comparator.kn5_both_visible.top1_recomputed", kn["top1_rows"],
      fit_seconds=kn["fit_seconds"], predict_seconds=kn["predict_seconds"], fit_info=kn["fit_info"])
    # bootstrap over evaluation documents
    ds_ev = data.ev_doc_start
    n_docs = int(ds_ev.shape[0] - 1)
    doc_t = np.searchsorted(ds_ev, data.targets, side="right") - 1
    n_by = np.bincount(doc_t, minlength=n_docs).astype(np.float64)
    hit_m = (main_sc["rank"] == 0).astype(np.float64)
    hit_a = (res["ablation"]["score"]["rank"] == 0).astype(np.float64)
    hit_l = (lb["pred"] == gold).astype(np.float64)
    hit_k = (kn["pred"] == gold).astype(np.float64)
    idx = protocol.bootstrap_index(n_docs, int(cfg["bootstrap"]["resamples"]))
    by = lambda h: np.bincount(doc_t, weights=h, minlength=n_docs)  # noqa: E731
    boot = {
        "cvtm_masked_top1": doc_bootstrap(by(hit_m), n_by, idx),
        "s_only_top1": doc_bootstrap(by(hit_a), n_by, idx),
        "cvtm_minus_left_bigram": doc_bootstrap(by(hit_m - hit_l), n_by, idx),
        "cvtm_minus_kn5_both": doc_bootstrap(by(hit_m - hit_k), n_by, idx),
        "cvtm_minus_s_only": doc_bootstrap(by(hit_m - hit_a), n_by, idx),
        "cvtm_minus_cbow_knn_logged": doc_bootstrap(by(hit_m), n_by, idx, const=logged["cbow_knn_visible"]),
    }
    for k, v in boot.items():
        M(f"bootstrap.{k}", v, label="derived", note="document-block bootstrap over evaluation documents, ratio of sums, 2,000 resamples (protocol.bootstrap_index); CBOW-kNN is a fixed logged number (no per-target predictions)")
    top1 = float(res["main"]["summary"]["top1"])
    v = verdict(top1, logged)
    v["ci95_top1"] = boot["cvtm_masked_top1"]["ci95"]
    v["ci95_minus_left_bigram"] = boot["cvtm_minus_left_bigram"]["ci95"]
    v["ci95_minus_kn5_both"] = boot["cvtm_minus_kn5_both"]["ci95"]
    if SMOKE:
        v["verdict"] = "not judged (smoke: pseudo-evaluation on training documents)"
    M("H8_11.verdict", v, label="derived",
      rule="MET iff validation top-1 > left bigram true context AND > CBOW-kNN at their logged values (docs/13 A3 H8.11); KN-5 both sides reported")
    # saved arrays
    od = out_dir()
    np.savez_compressed(od / "per_target.npz", targets=data.targets, gold=gold, pred=main_sc["pred"], rank=main_sc["rank"],
                        p_gold=main_sc["p_gold"], s_only_pred=res["ablation"]["score"]["pred"],
                        s_only_rank=res["ablation"]["score"]["rank"], s_only_p_gold=res["ablation"]["score"]["p_gold"],
                        left_bigram_pred=lb["pred"], kn5_both_pred=kn["pred"], doc=doc_t)
    np.savez_compressed(od / "weights.npz", cvtm_masked_weights=res["main"]["weights"],
                        cvtm_masked_weights_em=res["main"]["fit"]["weights"], cvtm_masked_trace=res["main"]["fit"]["trace"],
                        cvtm_masked_rows_per_param=res["main"]["fit"]["rows_per_param"],
                        s_only_weights=res["ablation"]["weights"], s_only_trace=res["ablation"]["fit"]["trace"],
                        s_only_rows_per_param=res["ablation"]["fit"]["rows_per_param"],
                        patterns=np.array(MK.PATTERN_NAMES), logn_levels=np.int64(MK.LOGN_LEVELS))
    M("outputs", {"per_target": "per_target.npz", "weights": "weights.npz"})
    ta1 = test_access_count()
    M("test_accesses_after", ta1)
    check("test_split_not_accessed", ta1 == ta0, before=ta0, after=ta1)
    M("host_peak_rss_gib", rss_peak_gib())
    M("wall_s_total", time.time() - T0)
    M("checks_failed", list(FAILED))
    log(f"H8.11: top-1 {top1:.6f} ({v['verdict']}); left bigram {logged['left_bigram_truectx']:.6f}, CBOW-kNN "
        f"{logged['cbow_knn_visible']:.6f}, KN-5 both {logged['kn5_both_visible']:.6f}; failed checks {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
