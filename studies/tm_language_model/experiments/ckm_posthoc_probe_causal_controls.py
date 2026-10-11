"""CKM causal probe, CONTROLS for gate G1 (EXPLORATORY: not pre-registered, no threshold of docs/13 is tested
here; validation split only, the test split is never loaded). Extends experiments/ckm_posthoc_probe_causal.py
(imported, unchanged; its run is results/20260924T012544Z_ckm_probe_causal_exploratory) with two controls:

(A) C0 control, "does Tsetlin LEARNING matter, or only having extra clause keys?"
    A YOUNG bank is trained from scratch by a fresh CudaEngine with the CP_anchor engine config (the epoch-4
    checkpoint's own EngineConfig: causal, V 8000, C 1,048,576, 13 bags, seed 1) until the bank is full: the
    first batch after which alive >= 0.98 C (hard cap --young-max-batches). The exact number of batches, examples
    and TM feedback events the young bank received is logged. The young bank then gets the SAME count read-out
    as the trained bank (same 5M-token prefix, same held-out documents, same validation positions), and the
    trained bank's rows are recomputed in this run for a paired comparison.
    Design choice: the young bank is trained on the FULL train split in the epoch-0 batch order of seed 1, i.e.
    it replays the anchor's own first B batches (the engine is counter-based and deterministic). The replay is
    checked against the anchor's logged diagnostics every 2,000 batches. So "young" and "trained" differ ONLY in
    the amount of Tsetlin training (B batches vs 4 epochs), not in data, seed, order or configuration. Training
    the young bank on the 5M prefix instead would take about 2.5 passes over the counting documents to fill the
    bank and would differ from the anchor in data as well.
    The 2,000-batch cap of the specification is not used as the stop rule: the anchor log gives seeded minus recycled
    = 365,851 clauses (0.35 C) after 2,000 batches [derived from results/20260923T042350Z_CP_anchor/log.txt], so
    the bank would be a third of the size, which is not an equal-size control. The measured alive count at batch
    2,000 is logged next to the stop point.

(B) Scale point: the trained bank's best rows with counts over a larger training prefix (default 20M tokens,
    falling back to 15M or 10M when the projected wall time or host memory does not fit; the choice is logged).
    The held-out documents that fit every mixture weight are the SAME 54 documents at every N (those of the 5M
    run); they are disjoint from the counting documents at every N. KN comparators are refitted on the SAME
    counting documents at every N.

Only the normalized linear mixture ("lin", bucket weights by L-BFGS on held-out rows) is rerun, with induction
and the unigram document cache, and with KN-5 as one more expert (the rows of the causal probe that matter).
Memory changes relative to the causal probe (results unchanged): the prefix fired lists are streamed (induction
statistics are accumulated chunk by chunk in position order, only the held-out rows' fired lists are kept),
count keys use 2^17-clause ranges, and the host copy of the count table is dropped after it is on the GPU.

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMLM_CPU_SLOTS=1
    source env/env.sh
    scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --limit-gib 60 -- \\
        scripts/run_logged.py --name ckm_probe_causal_controls_exploratory --config <scratch json> -- \\
        .venv-tm/bin/python -u experiments/ckm_posthoc_probe_causal_controls.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from engine_common import ROOT
from ckm_posthoc_probe import (k_induction_candidates, k_lin_obj, k_row_argmax, k_vote_scores,  # noqa: E402
                               merge_pairs, pairs_from_matrix, rowptr_of)
from ckm_posthoc_probe_causal import (CP_DIR, LN2, LOGN_CAP, NG_ORDERS, NINC_CAP, Gpu, csr_rows,  # noqa: E402
                                      doc_boot, doc_lo_of, k_cache_gold, k_cache_rows, k_gold_eval_log,
                                      ngram_level_ids)

sys.path.insert(0, str(ROOT / "src"))
CORPUS = "wikitext103"
EPOCH = 4                      # irrelevant in causal mode (tm_hidden is 0), as in the causal probe
OLD_RUN = ROOT / "results" / "20260924T012544Z_ckm_probe_causal_exploratory"
LIT_EMPTY = 0xFFFFFFFF
T0 = time.time()
METRICS: dict = {}
ROWS: dict = {}
PER_TOK: dict = {}
ANCHOR_LINE = re.compile(r"epoch 1 batch (\d+)/\d+: \d+ ex/s (cand/ex \S+ fired/ex \S+ events/ex \S+ seeded \d+ recycled \d+)")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def py(v):
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    if isinstance(v, np.ndarray):
        return py(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def M(name: str, value, label: str = "measured", **extra) -> None:
    METRICS[name] = {"value": py(value), "label": label, "exploratory": True, **py(extra)}


def flush_metrics() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


def proc_status_gib(key: str) -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def reset_hwm() -> None:
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def cgroup_ws_gib() -> float:
    try:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        inact = 0
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("inactive_file "):
                inact = int(line.split()[1])
        return (cur - inact) / 2 ** 30
    except OSError:
        return float("nan")


def mem_note(where: str) -> None:
    log(f"memory at {where}: rss {proc_status_gib('VmRSS'):.1f} GiB, peak since reset {proc_status_gib('VmHWM'):.1f} GiB, "
        f"cgroup working set {cgroup_ws_gib():.1f} GiB")


def save_outputs(X) -> None:
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if not run_dir:
        return
    np.savez_compressed(Path(run_dir) / "per_token_bits.npz", positions=np.arange(X["n_va"]), doc=X["doc_of_val"], **PER_TOK)
    Path(run_dir, "rows.json").write_text(json.dumps(py(ROWS), indent=1) + "\n")


# ------------------------------------------------------------------ CPU kernel (new here)
@njit(cache=False)
def k_ind_stats_chunk(ptr, ids, lo, ng, doc_lo, tok, is_cnt, last, tries, hits):
    """Streaming form of ckm_posthoc_probe_causal.k_ind_stats_skip: one chunk of prefix positions [lo, lo + n) in
    position order, fired lists as a chunk-local CSR. Called on consecutive chunks it gives exactly the same
    tries and hits as the one-shot kernel (held-out documents are skipped entirely)."""
    n = ptr.shape[0] - 1
    for a in range(n):
        p = lo + a
        if is_cnt[p] == 0:
            continue
        dl = doc_lo[p]
        y = tok[p]
        for e in range(ptr[a], ptr[a + 1]):
            k = ids[e]
            q = last[k]
            if q >= dl:
                tries[k] += 1
                if tok[q] == y:
                    hits[k] += 1
            last[k] = p
        for t in range(ng.shape[1]):
            k = ng[p, t]
            if k >= 0:
                q = last[k]
                if q >= dl:
                    tries[k] += 1
                    if tok[q] == y:
                        hits[k] += 1
                last[k] = p


# ------------------------------------------------------------------ engine passes
def forward_stream(eng, n: int, chunk: int, label: str, *, C: int, V: int, rbits: int, tok_dev=None, cnt_dev=None,
                   on_chunk=None, scores_fn=None):
    """Fired lists of positions 0..n-1 through CudaEngine._forward (the batched forward of predict_scores;
    read-only), chunk by chunk in position order. Per chunk: a chunk-local CSR (ids ascending per position) goes to
    on_chunk; count keys of the counting positions go to per-range host lists (if tok_dev is given)."""
    cp = eng.cp
    n_ranges = (C + (1 << rbits) - 1) >> rbits
    key_parts = [[] for _ in range(n_ranges)] if tok_dev is not None else None
    n_cand = 0
    n_fired = 0
    t0 = time.time()
    n_chunks = (n + chunk - 1) // chunk
    step = max(1, n_chunks // 10)
    for ci, lo in enumerate(range(0, n, chunk)):
        hi = min(n, lo + chunk)
        ids = cp.arange(lo, hi, dtype=cp.uint32)
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, ids, None, 0)
        eng._marks = []
        n_cand += int(fw.n_candidates)
        fe, fj = fw.fired_ex, fw.fired_j.astype(cp.uint32)
        if fe.size:
            assert bool(cp.all(fe[1:] >= fe[:-1])), "fired_ex is not example-major"
            order = cp.argsort((fe.astype(cp.uint64) << cp.uint64(32)) | fj.astype(cp.uint64))
            fe, fj = fe[order], fj[order]
            cnts = cp.bincount(fe.astype(cp.int64), minlength=hi - lo)
        else:
            cnts = cp.zeros(hi - lo, dtype=cp.int64)
        ptr_c = np.zeros(hi - lo + 1, np.int64)
        ptr_c[1:] = np.cumsum(cnts.get())
        fj_h = fj.get()
        n_fired += int(fj_h.shape[0])
        if key_parts is not None and fe.size:
            p = fe.astype(cp.int64) + lo
            keep = cnt_dev[p] == 1
            y = tok_dev[p].astype(cp.uint32)
            key = (fj & cp.uint32((1 << rbits) - 1)) * cp.uint32(V) + y
            rng = fj >> cp.uint32(rbits)
            for r in range(n_ranges):
                sel = keep & (rng == r)
                key_parts[r].append(key[sel].get())
            del p, keep, y, key, rng
        if on_chunk is not None:
            on_chunk(lo, hi, ptr_c, fj_h)
        if scores_fn is not None:
            scores_fn(lo, hi, fw.scores)
        del fw, fe, fj
        if (ci + 1) % step == 0:
            el = time.time() - t0
            log(f"  {label}: {hi}/{n} positions, {hi / el:.0f} pos/s, fired/pos {n_fired / hi:.1f}, cand/pos {n_cand / hi:.0f}")
    eng.synchronize()
    return key_parts, n_cand, n_fired, time.time() - t0


def gpu_counts_lean(key_parts, V: int, C: int, rbits: int):
    """Exact (clause, target) counts from the per-range 32-bit keys: radix sort + run lengths on the GPU, one range
    at a time (the host keys of a range are released as soon as it is on the device). Returns the CSR over the C
    clauses (indptr int64, idx uint16 ascending per clause, val int32), N per clause and the event total."""
    import cupy as cp
    pool = cp.get_default_memory_pool()
    D = np.zeros(C, np.int64)
    N = np.zeros(C, np.int64)
    idx_parts, val_parts = [], []
    total = 0
    for r in range(len(key_parts)):
        parts = key_parts[r]
        key_parts[r] = None
        lo_j = r << rbits
        n_j = min(1 << rbits, C - lo_j)
        if not parts:
            continue
        k = np.concatenate(parts)
        parts.clear()
        del parts
        total += int(k.shape[0])
        if k.shape[0] == 0:
            continue
        d = cp.asarray(k)
        del k
        d.sort()
        brk = cp.flatnonzero(d[1:] != d[:-1]) + 1
        starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
        del brk
        uk = d[starts]
        cnt = cp.diff(cp.concatenate([starts, cp.asarray([int(d.shape[0])], dtype=starts.dtype)]))
        del d, starts
        jl = (uk // cp.uint32(V)).astype(cp.int64)
        D[lo_j:lo_j + n_j] = cp.bincount(jl, minlength=n_j).get()
        N[lo_j:lo_j + n_j] = cp.rint(cp.bincount(jl, weights=cnt.astype(cp.float64), minlength=n_j)).astype(cp.int64).get()
        idx_parts.append((uk % cp.uint32(V)).astype(cp.uint16).get())
        val_parts.append(cnt.astype(cp.int32).get())
        del uk, cnt, jl
        pool.free_all_blocks()
    indptr = np.zeros(C + 1, np.int64)
    indptr[1:] = np.cumsum(D)
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.uint16)
    del idx_parts
    val = np.concatenate(val_parts) if val_parts else np.zeros(0, np.int32)
    del val_parts
    return indptr, idx, val, N, total


def bank_summary(st, V: int) -> dict:
    """Static description of a bank (alive clauses): included literals by kind, contiguous left token context
    length, clause age, vote slot use. Decoding as in the static audit (lit_id: segment << 17 | symbol << 1 | neg)."""
    alive = st["alive"].astype(bool)
    lid = st["lit_id"][alive]
    lst = st["lit_state"][alive]
    inc = (lst >= 128) & (lid != LIT_EMPTY)
    seg = (lid >> 17).astype(np.int64)
    sym = ((lid >> 1) & 0xFFFF).astype(np.int64)
    neg = (lid & 1).astype(bool)
    exact = seg < 16
    tokk = sym < V
    clus = (sym >= V) & (sym < V + 16 + 256 + 4096)
    n_inc = inc.sum(1)
    tot = max(1, int(inc.sum()))
    ptok = inc & exact & ~neg & tokk
    off = np.where(seg < 8, seg - 8, seg - 7)
    ng = np.zeros(lid.shape[0], np.int64)
    for o in range(1, 9):
        has = (ptok & (off == -o)).any(1)
        ng = np.where((ng == o - 1) & has, o, ng)
    step = int(st["step"][0])
    age = (np.uint64(step) - st["birth_step"][alive].astype(np.uint64)) & np.uint64(0xFFFFFFFF)
    vc = st["vote_cls"][alive]
    used = (vc != 0xFFFF).sum(1)
    return {
        "step": step, "alive": int(alive.sum()),
        "n_inc_mean": float(n_inc.mean()), "n_inc_hist_0_to_12": np.bincount(n_inc, minlength=13)[:13].tolist(),
        "included_literal_share": {
            "exact_positive_token": float((ptok).sum() / tot),
            "exact_positive_cluster": float((inc & exact & ~neg & clus).sum() / tot),
            "exact_negated": float((inc & exact & neg).sum() / tot),
            "bag_positive": float((inc & ~exact & ~neg).sum() / tot),
            "bag_negated": float((inc & ~exact & neg).sum() / tot)},
        "clauses_with_no_positive_exact_token_literal": float((ptok.sum(1) == 0).mean()),
        "left_contiguous_token_context_len_hist_0_to_8": np.bincount(ng, minlength=9).tolist(),
        "clause_age_steps_quantiles_10_50_90": np.quantile(age.astype(np.float64), [0.1, 0.5, 0.9]).tolist(),
        "used_vote_slots_mean": float(used.mean()),
    }


def train_young(cfg, full_c, args, out_dir: Path):
    """Fresh CudaEngine, CP_anchor config and seed, full train split, epoch-0 batch order; stop at the first batch
    after which alive >= young_fill * C (or at young_max_batches). Returns (state, state_hash, checkpoint path)."""
    import cupy as cp
    from tmlm.common.locks import yield_gpu_if_requested
    from tmlm.engine.checkpoint import load_checkpoint, save_checkpoint
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    anchor = {}
    for line in (CP_DIR / "log.txt").read_text().splitlines():
        m = ANCHOR_LINE.search(line)
        if m:
            anchor.setdefault(int(m.group(1)), m.group(2))   # first occurrence = epoch 1
    C = int(cfg.n_clauses)
    target = int(math.ceil(args.young_fill * C))
    M("young.stop_rule", f"first batch after which alive >= {args.young_fill} C = {target}, hard cap {args.young_max_batches} batches")
    t0 = time.time()
    eng = CudaEngine(cfg, full_c)
    eng.synchronize()
    M("young.engine_build_s", time.time() - t0, note="full train split (the anchor's own corpus), fresh state")
    M("young.batches_per_epoch", eng.n_batches())
    mem_note("young engine built")
    diag: dict[str, int] = {}
    trace = []
    checks = []
    stop_reason = "hard cap"
    done = 0
    alive = 0
    t1 = time.time()
    for b in range(args.young_max_batches):
        d = eng.train_batch(0, b)
        for k, v in d.items():
            diag[k] = diag.get(k, 0) + v
        done = b + 1
        alive = int(eng.d["alive"].sum().get())
        if done <= 4 or done % 250 == 0:
            trace.append([done, alive])
        if done % 2000 == 0:
            n_ex = diag["n_examples"]
            s = (f"cand/ex {diag['n_candidates'] / n_ex:.1f} fired/ex {diag['n_fired'] / n_ex:.2f} events/ex {diag['n_events_applied'] / n_ex:.2f} "
                 f"seeded {diag['n_seeded']} recycled {diag['n_recycled']}")
            ref = anchor.get(done)
            checks.append({"batch": done, "this_run": s, "anchor_log": ref, "equal": ref == s, "alive": alive})
            log(f"young batch {done}: {n_ex / (time.time() - t1):.0f} ex/s {s} alive {alive} ({alive / C:.4f} C) | anchor log: {ref} | "
                f"{'EQUAL' if ref == s else 'DIFFERENT'}")
            M("young.replay_checks", checks, note="cumulative epoch-1 diagnostics vs the same line of results/20260923T042350Z_CP_anchor/log.txt")
            flush_metrics()
        if done % 500 == 0:
            eng.synchronize()
            if yield_gpu_if_requested():
                log("yielded the GPU to an exclusive job and took the shared lock back")
        if alive >= target:
            stop_reason = f"alive >= {args.young_fill} C"
            break
    eng.synchronize()
    wall = time.time() - t1
    trace.append([done, alive])
    n_ex = diag["n_examples"]
    M("young.batches", done)
    M("young.examples_of_tm_feedback", n_ex, note="training examples (target positions) that went through the forward pass and feedback")
    M("young.epoch_fraction", n_ex / eng.n_examples, label="derived", note="examples / train-split examples (135,246,414)")
    M("young.stop_reason", stop_reason)
    M("young.alive_at_stop", alive)
    M("young.alive_fraction_at_stop", alive / C, label="derived")
    M("young.alive_trace", trace, note="[batches done, alive clauses]")
    M("young.diag_totals", diag, note="engine DIAG counters summed over all young batches (n_events_applied = clause events that got TM feedback)")
    M("young.train_wall_s", wall, note="shared GPU (a Track A E1 cell runs next to it)")
    M("young.examples_per_s", n_ex / wall)
    M("young.replay_checks_all_equal", bool(checks) and all(c["equal"] for c in checks), note=f"{len(checks)} checkpoints of 2,000 batches")
    flush_metrics()
    log(f"young bank: {done} batches, {n_ex} examples, alive {alive} ({alive / C:.4f} C), {stop_reason}, {wall:.0f} s")
    t0 = time.time()
    path = out_dir / "young_bank_checkpoint.npz"
    h = save_checkpoint(path, eng, 0, done, extra={"purpose": "C0 control of ckm_probe_causal_controls_exploratory", "stop_reason": stop_reason},
                        compress=True)
    del eng
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    cfg2, st, meta = load_checkpoint(path)
    assert meta["state_hash"] == h and cfg2 == cfg
    M("young.checkpoint", {"path": str(path), "state_hash": h, "bytes": path.stat().st_size, "write_and_reload_s": time.time() - t0},
      note="reloaded and hash-verified; the probe uses the reloaded state")
    flush_metrics()
    return st, h, path


# ------------------------------------------------------------------ one probe phase (bank x prefix size)
def run_phase(tag: str, st, cfg, full, n_tokens: int, X: dict, args, *, trained: bool, comparators: bool, ngram_rows: str) -> dict:
    """ngram_rows: 'all' (every n-gram row, and KN-5 + n-gram rows), 'check' (ngram.lin only, as a check against the
    phase that has 'all'), 'none'."""
    import cupy as cp
    from scipy.optimize import minimize
    from tmlm.decode.bank import class_bias_from_counts
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    from tmlm.eval.baselines.ngram import KNModel
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import contiguous

    reset_hwm()
    tph = time.time()
    TT: dict = {}
    V, C = int(cfg.vocab), int(cfg.n_clauses)
    pool = cp.get_default_memory_pool()
    tok_va, lo_va, n_va, n_va_tok, n_bytes = X["tok_va"], X["lo_va"], X["n_va"], X["n_va_tok"], X["n_bytes"]

    def pm(name, value, label="measured", **kw):
        M(f"{tag}.{name}", value, label, **kw)

    log(f"==================== phase {tag}: {'trained' if trained else 'young'} bank, prefix {n_tokens} tokens")
    tr = contiguous(slice_train(full, n_tokens))
    n_tr = int(tr.tok.shape[0])
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    lo_tr = doc_lo_of(ds_tr, n_tr)
    n_doc = ds_tr.shape[0] - 1
    ds_small = X["ds_small"]
    assert n_doc >= ds_small.shape[0] - 1 and np.array_equal(ds_tr[:ds_small.shape[0]], ds_small), "prefix does not extend the small prefix"
    ho_docs = X["ho_docs"]
    doc_of_tr = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_ho = np.isin(doc_of_tr, ho_docs)
    del doc_of_tr
    cnt_pos = np.flatnonzero(~is_ho)
    ho_pos = np.flatnonzero(is_ho)
    pm("bank", "trained (CP_anchor epoch 4)" if trained else "young (CP_anchor config, fresh, stopped at fill)")
    pm("data.prefix_tokens", n_tr, note="first tokens of WikiText-103 train at V 8000, cut at a document boundary")
    pm("data.prefix_docs", n_doc)
    pm("split.heldout_docs", int(ho_docs.shape[0]), note="the same held-out documents at every prefix size (those of the 5M run)")
    pm("split.counting_positions", int(cnt_pos.shape[0]))
    pm("split.heldout_positions", int(ho_pos.shape[0]))

    # ---------------------------------------------------------------- n-gram components (left-1 .. left-5), before the forward pass
    t0 = time.time()
    y_cnt = tok_tr[cnt_pos]
    levels = ngram_level_ids(tok_tr, lo_tr, tok_va[:n_va], lo_va[:n_va], NG_ORDERS, V)
    ng_ptr, ng_idx, ng_val, ng_N, ng_D, ng_ninc = [], [], [], [], [], []
    ng_tr = np.full((n_tr, NG_ORDERS), -1, np.int64)
    ng_va = np.full((n_va, NG_ORDERS), -1, np.int64)
    base = C
    run_e = 0
    for k in range(len(levels)):
        ia, ib = levels[k]
        u, cnt = np.unique(ia[cnt_pos] * V + y_cnt, return_counts=True)
        ctx = u // V
        tg = (u % V).astype(np.uint16)
        ckeys, start = np.unique(ctx, return_index=True)
        indptr_k = np.append(start, u.shape[0]).astype(np.int64)
        ng_ptr.append(indptr_k[1:] + run_e)
        run_e += int(u.shape[0])
        ng_idx.append(tg)
        ng_val.append(cnt.astype(np.int32))
        ng_N.append(np.add.reduceat(cnt, start).astype(np.int64))
        ng_D.append(np.diff(indptr_k))
        ng_ninc.append(np.full(ckeys.shape[0], k + 1, np.int64))
        for ids_, out in ((ia, ng_tr), (ib, ng_va)):
            j = np.searchsorted(ckeys, ids_)
            jc = np.minimum(j, ckeys.shape[0] - 1)
            found = ckeys[jc] == ids_
            out[found, k] = base + jc[found]
        pm(f"ngram.contexts_left{k + 1}", int(ckeys.shape[0]))
        base += ckeys.shape[0]
        levels[k] = None
        del u, cnt, ctx, ckeys, start, ia, ib
    del levels
    n_comp = base
    TT["ngram_s"] = time.time() - t0
    log(f"{tag}: n-gram tables {n_comp - C} contexts, {run_e} entries ({TT['ngram_s']:.0f} s)")

    # ---------------------------------------------------------------- prefix forward pass (streamed)
    t0 = time.time()
    eng = CudaEngine(cfg, tr)
    eng.load_state(st)
    eng.synchronize()
    pm("engine.prefix_build_s", time.time() - t0)
    tok_dev = cp.asarray(tok_tr.astype(np.uint16))
    cnt_dev = cp.asarray((~is_ho).astype(np.uint8))
    is_cnt = (~is_ho).astype(np.uint8)
    last = np.full(n_comp, -1, np.int64)
    tries = np.zeros(n_comp, np.int64)
    hits = np.zeros(n_comp, np.int64)
    rs = np.random.default_rng(12345)
    chk = np.sort(rs.choice(n_tr, size=100, replace=False))
    chk_lists: dict = {}
    ho_lens, ho_ids_parts = [], []
    t_ind = [0.0]

    def on_chunk(lo, hi, ptr_c, fj_h):
        t1 = time.time()
        k_ind_stats_chunk(ptr_c, fj_h, lo, ng_tr, lo_tr, tok_tr, is_cnt, last, tries, hits)
        t_ind[0] += time.time() - t1
        sel = np.flatnonzero(is_ho[lo:hi])
        if sel.size:
            rp, ids = csr_rows(ptr_c, fj_h, sel)
            ho_lens.append(np.diff(rp))
            ho_ids_parts.append(ids)
        a, b = np.searchsorted(chk, [lo, hi])
        for p in chk[a:b].tolist():
            chk_lists[p] = fj_h[ptr_c[p - lo]:ptr_c[p - lo + 1]].copy()

    log(f"{tag}: prefix forward pass (causal, every position a target; counts, induction statistics and held-out rows streamed)")
    key_parts, cand_tr, fired_tr, wall_tr = forward_stream(eng, n_tr, args.chunk, f"{tag} prefix", C=C, V=V, rbits=args.rbits,
                                                           tok_dev=tok_dev, cnt_dev=cnt_dev, on_chunk=on_chunk)
    TT["prefix_forward_s"] = wall_tr
    key_bytes = sum(int(a.nbytes) for parts in key_parts for a in parts)
    pm("throughput.prefix_forward_pos_per_s", n_tr / wall_tr, note="CudaEngine._forward + fired-list download + streamed induction statistics; shared GPU (a Track A E1 cell runs next to it)")
    pm("throughput.induction_stats_s", t_ind[0])
    pm("prefix.fired_per_position", fired_tr / n_tr)
    pm("prefix.candidates_per_position", cand_tr / n_tr)
    for i, p in enumerate(chk.tolist()):
        a = eng.fired_clauses(EPOCH, p)
        assert np.array_equal(a, chk_lists[p]), f"prefix fired list differs from the API at {p}"
        if i < 3:
            b = eng.fired_clauses_bruteforce(EPOCH, p)
            assert np.array_equal(np.sort(b), chk_lists[p]), f"prefix fired list differs from brute force at {p}"
    pm("check.prefix_fired_vs_api", "100 random positions equal to fired_clauses, 3 equal to fired_clauses_bruteforce")
    del eng, tok_dev, cnt_dev, chk_lists
    pool.free_all_blocks()
    ho_rp = np.zeros(ho_pos.shape[0] + 1, np.int64)
    ho_rp[1:] = np.cumsum(np.concatenate(ho_lens)) if ho_lens else 0
    ho_ids = np.concatenate(ho_ids_parts) if ho_ids_parts else np.zeros(0, np.uint32)
    del ho_lens, ho_ids_parts
    assert ho_rp[-1] == ho_ids.shape[0]
    mem_note(f"{tag} after the prefix pass (count keys {key_bytes / 2 ** 30:.1f} GiB on the host)")
    t0 = time.time()
    indptr_e, idx_e, val_e, N_e, n_events = gpu_counts_lean(key_parts, V, C, args.rbits)
    del key_parts
    gc.collect()
    D_e = np.diff(indptr_e).astype(np.int64)
    assert int(N_e.sum()) == n_events and int(val_e.astype(np.int64).sum()) == n_events
    TT["counts_s"] = time.time() - t0
    csr_bytes = int(idx_e.nbytes + val_e.nbytes)
    pm("counts.events", int(n_events), note="(clause, next token) pairs over the counting positions")
    pm("counts.distinct_pairs", int(idx_e.shape[0]))
    pm("counts.clauses_with_counts", int((N_e > 0).sum()))
    pm("counts.N_median_of_counted", float(np.median(N_e[N_e > 0])))
    pm("counts.D_mean_of_counted", float(D_e[N_e > 0].mean()))
    pm("counts.gpu_sort_s", TT["counts_s"])
    log(f"{tag}: engine counts {n_events} events, {idx_e.shape[0]} distinct pairs, {(N_e > 0).sum()} clauses ({TT['counts_s']:.0f} s)")
    mem_note(f"{tag} after counting")
    flush_metrics()

    # ---------------------------------------------------------------- validation pass + pipeline checks
    t0 = time.time()
    va_lens, va_ids_parts = [], []
    acc = {"raw_hits": 0, "cal_hits": 0, "cal_top5": 0, "cal_nats": 0.0, "worst_norm": 0.0}
    eng_lp = np.zeros(n_va)
    cp_metrics = X["cp_metrics"]
    cal = cp_metrics["epoch4.calibration"]["value"]
    a_q16, b_q16 = int(cal["a_q16"]), int(cal["b_q16"])
    d_bias = cp.asarray(class_bias_from_counts(X["train_counts_full"], V).astype(np.int64))
    d_vtok = cp.asarray(tok_va.astype(np.int64))

    def on_chunk_v(lo, hi, ptr_c, fj_h):
        va_lens.append(np.diff(ptr_c))
        va_ids_parts.append(fj_h)

    def on_scores(lo, hi, scores):
        g = d_vtok[lo:hi]
        rows = cp.arange(hi - lo)
        acc["raw_hits"] += int((cp.argmax(scores, axis=1) == g).sum())
        if not trained:
            return
        num = scores.astype(cp.int64) * (a_q16 * 256) + (b_q16 * d_bias)[None, :]
        acc["cal_hits"] += int((cp.argmax(num, axis=1) == g).sum())
        ng_ = num[rows, g][:, None]
        rank = (num > ng_).sum(axis=1) + ((num == ng_) & (cp.arange(V)[None, :] < g[:, None])).sum(axis=1)
        acc["cal_top5"] += int((rank < 5).sum())
        lg = num.astype(cp.float64) / float(1 << 24)
        m = lg.max(axis=1, keepdims=True)
        z = lg - m
        lse = cp.log(cp.exp(z).sum(axis=1, keepdims=True))
        lp = z - lse
        acc["worst_norm"] = max(acc["worst_norm"], float(cp.abs(cp.log(cp.exp(lp).sum(axis=1))).max()))
        lpg = lp[rows, g]
        acc["cal_nats"] += float(-lpg.sum())
        eng_lp[lo:hi] = lpg.get()

    eng_v = CudaEngine(cfg, X["val_corpus"])
    eng_v.load_state(st)
    log(f"{tag}: validation forward pass")
    _, cand_va, fired_va, wall_va = forward_stream(eng_v, n_va, 8192, f"{tag} validation", C=C, V=V, rbits=args.rbits,
                                                   on_chunk=on_chunk_v, scores_fn=on_scores)
    ptr_va = np.zeros(n_va + 1, np.int64)
    ptr_va[1:] = np.cumsum(np.concatenate(va_lens))
    ids_va = np.concatenate(va_ids_parts)
    del va_lens, va_ids_parts
    top1_raw = acc["raw_hits"] / n_va
    pm("throughput.validation_forward_pos_per_s", n_va / wall_va)
    pm("validation.fired_per_position", fired_va / n_va)
    pm("validation.candidates_per_position", cand_va / n_va)
    pm("engine_signed_votes.raw_top1", top1_raw, note="argmax of the bank's own signed-vote class sums (ties to the lowest id)")
    c1_ok = True
    if trained:
        assert acc["worst_norm"] < 1e-6
        logged_raw = cp_metrics["epoch4.validation.causal_top1_raw_scores"]["value"]
        logged_bpb = cp_metrics["epoch4.validation.bits_per_byte"]["value"]
        pm("C1.logged_raw_top1", logged_raw, source="results/20260923T042350Z_CP_anchor/metrics.json")
        pm("C1.engine_calibrated_top1", acc["cal_hits"] / n_va)
        if n_va == n_va_tok:
            bpb = acc["cal_nats"] / LN2 / n_bytes
            pm("C1.engine_bits_per_byte_logged_calibration", bpb)
            c1_ok = bool(top1_raw == logged_raw and abs(bpb - logged_bpb) < 1e-9)
            pm("C1.passed", c1_ok)
            PER_TOK[f"{tag}.engine.signed_votes_logged_calibration"] = (-eng_lp / LN2).astype(np.float32)
            ROWS[f"{tag}.engine.signed_votes_logged_calibration"] = {
                "bits_per_byte": bpb, "bits_per_token": float((-eng_lp / LN2).mean()), "top1_raw": top1_raw,
                "top1": acc["cal_hits"] / n_va, "top5": acc["cal_top5"] / n_va,
                "note": "trained on all 135M train tokens for 4 epochs; calibration fitted on validation (logged)"}
        else:
            pm("C1.passed", None, label="unverified", note="validation subset (smoke): the logged values cover the full split")
    rows_chk = np.sort(rs.choice(n_va, size=min(2000, n_va), replace=False))
    s_api = eng_v.predict_scores(EPOCH, rows_chk.astype(np.uint32))
    rec = k_vote_scores(ptr_va, ids_va, rows_chk, st["vote_cls"], st["vote_w"].astype(np.int32), V)
    mism = int((rec != s_api).any(axis=1).sum())
    pm("C1.score_rows_mismatch_vs_predict_scores", mism, note="2000 random validation positions: vote sums rebuilt from our fired lists vs predict_scores")
    for i, p in enumerate(rows_chk[:100].tolist()):
        assert np.array_equal(eng_v.fired_clauses(EPOCH, p), ids_va[ptr_va[p]:ptr_va[p + 1]])
        if i < 3:
            assert np.array_equal(np.sort(eng_v.fired_clauses_bruteforce(EPOCH, p)), ids_va[ptr_va[p]:ptr_va[p + 1]])
    pm("check.validation_fired_vs_api", "100 positions equal to fired_clauses, 3 equal to fired_clauses_bruteforce")
    del eng_v, d_bias, d_vtok, s_api, rec
    pool.free_all_blocks()
    log(f"{tag}: signed-vote raw top-1 {top1_raw:.6f}; rebuilt-score mismatches {mism}; C1 {c1_ok} ({time.time() - t0:.0f} s)")
    flush_metrics()
    if mism != 0 or not c1_ok:
        raise RuntimeError(f"{tag}: pipeline check C1 failed (mismatches {mism}, C1 {c1_ok})")

    # ---------------------------------------------------------------- components (engine clauses + n-gram contexts)
    t0 = time.time()
    indptr = np.concatenate([indptr_e, np.concatenate(ng_ptr) + indptr_e[-1]])
    del ng_ptr
    idx = np.concatenate([idx_e] + ng_idx)
    del idx_e, ng_idx
    valc = np.concatenate([val_e] + ng_val)
    del val_e, ng_val
    N = np.concatenate([N_e] + ng_N)
    D = np.concatenate([D_e] + ng_D)
    del ng_N, ng_D
    alive = st["alive"].astype(bool)
    inc = (st["lit_state"] >= 128) & (st["lit_id"] != LIT_EMPTY) & alive[:, None]
    ninc_e = inc.sum(axis=1).astype(np.int64)
    del inc
    pm("engine.alive_clauses", int(alive.sum()))
    ninc = np.concatenate([ninc_e] + ng_ninc)
    fam = np.concatenate([np.zeros(C, np.int64), np.ones(n_comp - C, np.int64)])
    del ng_ninc
    assert indptr.shape[0] == n_comp + 1 and indptr[-1] == idx.shape[0]
    inv_N = np.where(N > 0, 1.0 / np.maximum(N, 1), 0.0)
    amax = k_row_argmax(indptr, idx, valc)
    uni = np.bincount(y_cnt, minlength=V).astype(np.float64)
    p_uni = (uni + 1.0) / (uni.sum() + V)
    assert abs(p_uni.sum() - 1.0) < 1e-12
    logn = np.minimum(np.floor(np.log2(np.maximum(N, 1))).astype(np.int64), LOGN_CAP)
    bucket = (fam * (NINC_CAP + 1) + np.minimum(ninc, NINC_CAP)) * (LOGN_CAP + 1) + logn
    n_buckets = 2 * (NINC_CAP + 1) * (LOGN_CAP + 1)
    has_counts = N > 0
    del logn, fam, D
    h_comp = np.where(tries > 0, hits / np.maximum(tries, 1), 0.0)
    pm("components.total", int(n_comp))
    pm("components.count_entries_total", int(idx.shape[0]))
    pm("induction.engine_hits_over_tries", float(hits[:C].sum() / max(1, tries[:C].sum())), note="counting positions, document order, exact most recent earlier firing")
    pm("induction.ngram_hits_over_tries", float(hits[C:].sum() / max(1, tries[C:].sum())))
    del last, tries, hits
    TT["components_s"] = time.time() - t0

    # ---------------------------------------------------------------- row sets
    t0 = time.time()
    va_pos = np.arange(n_va, dtype=np.int64)
    sets = {}
    for sname, rp_e, ids_e, pos, tok, lo, ng in (("val", ptr_va, ids_va, va_pos, tok_va, lo_va[:n_va], ng_va),
                                                  ("ho", ho_rp, ho_ids, ho_pos, tok_tr, lo_tr[ho_pos], ng_tr[ho_pos])):
        n = pos.shape[0]
        gold = tok[pos]
        rowid_all = np.repeat(np.arange(n, dtype=np.int64), np.diff(rp_e))
        comp_all = ids_e.astype(np.int64)
        keep = has_counts[comp_all]
        eng_pairs = (rowid_all[keep], comp_all[keep])
        del rowid_all, comp_all, keep
        ng_pairs = pairs_from_matrix(ng)
        arms = {"engine": eng_pairs, "ngram": ng_pairs, "union": merge_pairs(eng_pairs, ng_pairs)}
        s = dict(n=n, pos=pos, gold=gold, tok=tok, lo=lo, arms={})
        vis = np.ones(tok.shape[0], np.uint8)
        for arm, (rowid, comp) in arms.items():
            rp = rowptr_of(rowid, n)
            ts = k_induction_candidates(rp_e, ids_e, ng, pos, lo, vis, np.arange(n, dtype=np.int64), rp, comp, tok, n_comp)
            s["arms"][arm] = dict(rowid=rowid.astype(np.int64), comp=comp.astype(np.int64), rp=rp, m=np.diff(rp), ts=ts.astype(np.int64))
        pm(f"{sname}.engine_fired_per_row", float(rp_e[-1] / max(1, n)))
        pm(f"{sname}.engine_fired_with_counts_per_row", float(eng_pairs[0].shape[0] / max(1, n)))
        pm(f"{sname}.ngram_contexts_found_per_row", float(ng_pairs[0].shape[0] / max(1, n)))
        pm(f"{sname}.frac_rows_without_counted_engine_clause", float((np.diff(s["arms"]["engine"]["rp"]) == 0).mean()))
        del arms, eng_pairs, ng_pairs, vis
        sets[sname] = s
    del ho_ids, ids_va, ng_tr, ptr_va, ho_rp
    gc.collect()
    TT["row_sets_s"] = time.time() - t0
    log(f"{tag}: row sets and induction candidates ({TT['row_sets_s']:.0f} s)")

    # ---------------------------------------------------------------- GPU count table, gold counts per pair
    t0 = time.time()
    gpu = Gpu(V)
    gpu.csr["full"] = dict(indptr=cp.asarray(indptr), idx=cp.asarray(idx), val=cp.asarray(valc), evd=None)
    tok_va_dev = cp.asarray(tok_va.astype(np.uint16))
    for sname, s in sets.items():
        for arm, A in s["arms"].items():
            g = s["gold"][A["rowid"]]
            A["cnt_full"] = gpu.pair_count("full", A["comp"], g)
            if sname == "val":
                n = s["n"]
                in_sup = np.bincount(A["rowid"], weights=(A["cnt_full"] > 0), minlength=n) > 0
                am = np.bincount(A["rowid"], weights=(amax[A["comp"]] == g), minlength=n) > 0
                pm(f"oracle.{arm}.gold_in_union_of_supports", float(in_sup.mean()))
                pm(f"oracle.{arm}.some_component_argmax_is_gold", float(am.mean()))
            del g
    A = sets["val"]["arms"]["union"]
    samp = np.random.default_rng(7).choice(A["comp"].shape[0], size=min(20000, A["comp"].shape[0]), replace=False)
    g = sets["val"]["gold"][A["rowid"][samp]]
    for i, k in enumerate(A["comp"][samp].tolist()):
        seg = idx[indptr[k]:indptr[k + 1]]
        j = np.searchsorted(seg, g[i])
        ref = int(valc[indptr[k] + j]) if j < seg.shape[0] and seg[j] == g[i] else 0
        assert ref == A["cnt_full"][samp[i]], "GPU gold count differs from the CPU lookup"
    pm("check.gpu_gold_counts_vs_cpu", "20000 random validation union pairs equal")
    del idx, valc, amax, indptr
    gc.collect()
    TT["gold_counts_s"] = time.time() - t0
    log(f"{tag}: gold counts and oracles ({TT['gold_counts_s']:.0f} s)")
    mem_note(f"{tag} before the estimators")
    flush_metrics()

    # ---------------------------------------------------------------- estimators (lin only, as selected by the causal probe)
    sv = sets["val"]
    ho = sets["ho"]
    fit_rows = (np.arange(ho["n"]) % args.fit_every) == 0

    def arrays(sname, arm, induction):
        s = sets[sname]
        A = s["arms"][arm]
        comp = A["comp"]
        cnt = A["cnt_full"]
        inv = inv_N[comp]
        pk = cnt * inv
        if induction:
            ts = A["ts"]
            hh = np.where(ts >= 0, h_comp[comp], 0.0)
            pkfull = np.where(ts >= 0, hh * (ts == s["gold"][A["rowid"]]) + (1.0 - hh) * pk, pk)
        else:
            ts = np.full(comp.shape[0], -1, np.int64)
            hh = np.zeros(comp.shape[0])
            pkfull = pk
        return dict(rowid=A["rowid"], comp=comp, rp=A["rp"], m=A["m"], ts=ts, hh=hh, pkfull=pkfull, inv=inv)

    def weights(b, params, n):
        a = params["alpha"][bucket[b["comp"]]]
        Z = np.bincount(b["rowid"], weights=a, minlength=n) + params["a0"]
        return a / Z[b["rowid"]], params["a0"] / Z

    def sub_rows(b, keep_rows):
        newid = np.cumsum(keep_rows) - 1
        sel = keep_rows[b["rowid"]]
        rid = newid[b["rowid"][sel]]
        n2 = int(keep_rows.sum())
        rp = rowptr_of(rid, n2)
        return dict(rowid=rid, comp=b["comp"][sel], rp=rp, m=np.diff(rp), ts=b["ts"][sel], hh=b["hh"][sel],
                    pkfull=b["pkfull"][sel], inv=b["inv"][sel]), n2

    def fit(arm, induction):
        t1 = time.time()
        b0 = arrays("ho", arm, induction)
        b, n2 = sub_rows(b0, fit_rows)
        del b0
        pu_g = p_uni[ho["gold"][fit_rows]]
        bk = bucket[b["comp"]]
        used = np.unique(bk)

        def obj(theta):
            alpha = np.zeros(n_buckets)
            alpha[used] = np.exp(theta)
            ll, num, den, _, _ = k_lin_obj(b["rp"], b["pkfull"], bk, pu_g, alpha, 1.0)
            return -ll, -(alpha[used] * (num[used] - den[used]))
        res = minimize(obj, np.zeros(used.shape[0]), jac=True, method="L-BFGS-B", options={"maxiter": args.lin_iters})
        alpha = np.zeros(n_buckets)
        alpha[used] = np.exp(res.x)
        return {"alpha": alpha, "a0": 1.0}, {"heldout_ll": float(-res.fun), "converged": bool(res.success), "iters": int(res.nit),
                                             "n_params": int(used.shape[0]), "fit_rows": n2, "fit_s": time.time() - t1}

    def gold_bits(sname, arm, params, induction, mu=None):
        s = sets[sname]
        b = arrays(sname, arm, induction)
        w, wuni = weights(b, params, s["n"])
        with np.errstate(divide="ignore"):
            lw, lwu = np.log(w), np.log(wuni)
        lpm, pc, has, werr = k_gold_eval_log(b["rp"], lw, b["pkfull"], lwu, p_uni[s["gold"]], s["gold"], s["pos"], s["lo"], s["tok"], V)
        assert werr < 1e-9, f"weights do not sum to one: {werr}"
        if mu is None:
            lp = lpm
        else:
            with np.errstate(divide="ignore"):
                lp = np.where(has == 1, np.logaddexp(np.log1p(-mu) + lpm, np.log(mu) + np.log(pc)), lpm)
        assert np.all(np.isfinite(lp)), "a gold log probability is not finite"
        return lp, lpm, pc, has, b, w, wuni

    def bpb_of(bits_tok):
        return float(bits_tok.sum() / n_bytes) if n_va == n_va_tok else None

    def record(name, row, bits_tok):
        rname = f"{tag}.{name}"
        ROWS[rname] = row
        PER_TOK[rname] = bits_tok.astype(np.float32)
        for k in ("bits_per_byte", "bits_per_token", "top1", "top5", "heldout_bits_per_token"):
            if k in row:
                M(f"rows.{rname}.{k}", row[k])
        M(f"rows.{rname}.spec", {k: v for k, v in row.items() if k not in ("bits_per_byte", "bits_per_token", "top1", "top5", "heldout_bits_per_token")})
        flush_metrics()
        f = lambda k: (f"{row[k]:.4f}" if isinstance(row.get(k), float) else "n/a")  # noqa: E731
        log(f"{rname:52s} bpb {f('bits_per_byte')} bits/tok {f('bits_per_token')} top1 {f('top1')} top5 {f('top5')} held-out {f('heldout_bits_per_token')}")

    def evaluate(name, arm, params, induction, mu=None, fit_info=None):
        t1 = time.time()
        lp, _, _, _, b, w, wuni = gold_bits("val", arm, params, induction, mu)
        p = np.exp(lp)
        hh = b["hh"]
        coef = w * (1.0 - hh) * b["inv"]
        tcoef = w * hh
        out = gpu.mix("full", b["rp"], b["comp"], coef, b["ts"], tcoef, wuni, p_uni, 0, sv["gold"],
                      mu=(mu or 0.0), pos=sv["pos"], dl=sv["lo"], tok_dev=tok_va_dev)
        maxerr = float(np.abs(out["sum"] - 1.0).max())
        assert maxerr < 1e-9, f"{name}: a probability row does not sum to one ({maxerr})"
        rel = float(np.abs(out["pg"] - p).max())
        assert rel < 1e-9, f"{name}: GPU gold probability differs from the CPU path (max abs {rel})"
        bits_tok = -lp / LN2
        del b, w, wuni, coef, tcoef, p
        ho_lp = gold_bits("ho", arm, params, induction, mu)[0]
        row = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()), "top1": float((out["rank"] == 0).mean()),
               "top5": float((out["rank"] < 5).mean()), "heldout_bits_per_token": float((-ho_lp / LN2).mean()),
               "arm": arm, "estimator": "lin", "induction": bool(induction), "doc_cache_mu": mu, "max_row_sum_error": maxerr,
               "gpu_vs_cpu_gold_prob_max_abs_diff": rel, "eval_s": time.time() - t1}
        if fit_info is not None:
            row["fit"] = fit_info
        record(name, row, bits_tok)
        return row

    def fit_mu(arm, params, induction):
        _, lpm, pc, has, _, _, _ = gold_bits("ho", arm, params, induction)
        pm_ = np.exp(lpm)
        ok = (has == 1) & ((pm_ + pc) > 0)
        mu = 0.1
        for _ in range(200):
            mu = float((mu * pc[ok] / ((1 - mu) * pm_[ok] + mu * pc[ok])).mean())
        return mu

    t0 = time.time()
    arms_run = ["engine", "ngram", "union"] if ngram_rows != "none" else ["engine", "union"]
    for arm in arms_run:
        p0, i0 = fit(arm, False)
        evaluate(f"{arm}.lin", arm, p0, False, fit_info=i0)
        if arm == "ngram" and ngram_rows == "check":
            continue
        p1, i1 = fit(arm, True)
        evaluate(f"{arm}.lin+induction", arm, p1, True, fit_info=i1)
        for ind, params in ((False, p0), (True, p1)):
            mu = fit_mu(arm, params, ind)
            evaluate(f"{arm}.lin+{'induction+' if ind else ''}doc_cache", arm, params, ind, mu=mu)
    TT["lin_rows_s"] = time.time() - t0

    # ---------------------------------------------------------------- KN-5 on the counting documents; KN-5-anchored mixtures
    t0 = time.time()
    keep_docs = np.setdiff1d(np.arange(n_doc), ho_docs)

    def docs_stream(docs):
        seg = [tok_tr[ds_tr[d]:ds_tr[d + 1]] for d in docs]
        ds = np.zeros(len(seg) + 1, np.int64)
        ds[1:] = np.cumsum([x.shape[0] for x in seg])
        return np.concatenate(seg).astype(np.int64), ds
    tk, dk = docs_stream(keep_docs)
    th, dh = docs_stream(ho_docs)
    assert np.array_equal(th, tok_tr[ho_pos]), "held-out stream order differs from the held-out rows"
    kn = KNModel(5, V, discount_mode="modified").fit(tk, dk)
    TT["kn_fit_s"] = time.time() - t0
    pm("kn.fit_s", TT["kn_fit_s"])
    vt, vds = tok_va[:n_va], np.append(X["ds_va"][X["ds_va"] < n_va], n_va)
    kn_lp2 = {"val": kn.logprob(vt, vds, order=5), "ho": kn.logprob(th, dh, order=5)}

    def kn_bundle(sname, arm, induction, keep=None):
        s = sets[sname]
        b = arrays(sname, arm, induction)
        n = s["n"]
        pkn = np.exp2(kn_lp2[sname])
        gold, pos, lo = s["gold"], s["pos"], s["lo"]
        if keep is not None:
            b, n = sub_rows(b, keep)
            pkn, gold, pos, lo = pkn[keep], gold[keep], pos[keep], lo[keep]
        rowid = np.concatenate([b["rowid"], np.arange(n)])
        pk = np.concatenate([b["pkfull"], pkn])
        bk = np.concatenate([bucket[b["comp"]], np.full(n, n_buckets)])
        o = np.argsort(rowid, kind="stable")
        return dict(rowid=rowid[o], pk=pk[o], bk=bk[o], rp=rowptr_of(rowid[o], n), n=n, gold=gold, pos=pos, lo=lo, tok=s["tok"])

    def kn_eval(bd, alpha):
        a = alpha[bd["bk"]]
        Z = np.bincount(bd["rowid"], weights=a, minlength=bd["n"]) + 1.0
        with np.errstate(divide="ignore"):
            lw = np.log(a / Z[bd["rowid"]])
        lwu = np.log(1.0 / Z)
        lpm, pc, has, werr = k_gold_eval_log(bd["rp"], lw, bd["pk"], lwu, p_uni[bd["gold"]], bd["gold"], bd["pos"], bd["lo"], bd["tok"], V)
        assert werr < 1e-9
        return lpm, pc, has

    def with_cache(lpm, pc, has, mu):
        with np.errstate(divide="ignore"):
            return np.where(has == 1, np.logaddexp(np.log1p(-mu) + lpm, np.log(mu) + np.log(pc)), lpm)

    kn5_arms = (["ngram"] if ngram_rows == "all" else []) + ["union"] + (["engine"] if args.kn5_engine else [])
    for arm in kn5_arms:
        for ind in (False, True):
            t1 = time.time()
            bf = kn_bundle("ho", arm, ind, fit_rows)
            used = np.unique(bf["bk"])
            pu_g = p_uni[bf["gold"]]

            def obj(theta):
                alpha = np.zeros(n_buckets + 1)
                alpha[used] = np.exp(theta)
                ll, num, den, _, _ = k_lin_obj(bf["rp"], bf["pk"], bf["bk"], pu_g, alpha, 1.0)
                return -ll, -(alpha[used] * (num[used] - den[used]))
            res = minimize(obj, np.zeros(used.shape[0]), jac=True, method="L-BFGS-B", options={"maxiter": args.lin_iters})
            alpha = np.zeros(n_buckets + 1)
            alpha[used] = np.exp(res.x)
            del bf
            bh = kn_bundle("ho", arm, ind)
            lpm_h, pc_h, has_h = kn_eval(bh, alpha)
            del bh
            bv = kn_bundle("val", arm, ind)
            lpm_v, pc_v, has_v = kn_eval(bv, alpha)
            del bv
            assert np.all(np.isfinite(lpm_v))
            name = f"kn5+{arm}.lin" + ("+induction" if ind else "")
            info = {"heldout_ll": float(-res.fun), "converged": bool(res.success), "iters": int(res.nit), "n_params": int(used.shape[0]),
                    "fit_rows": int(fit_rows.sum()), "kn_expert_weight_over_a0": float(alpha[n_buckets])}
            bits_tok = -lpm_v / LN2
            record(name, {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                          "heldout_bits_per_token": float((-lpm_h / LN2).mean()), "arm": arm, "estimator": "lin with a KN-5 expert",
                          "induction": ind, "top1": None, "top5": None, "fit": info, "eval_s": time.time() - t1,
                          "note": "bits only (the dense KN-5 rows are not combined with the clause rows on the GPU)"}, bits_tok)
            ok = (has_h == 1)
            pmh = np.exp(lpm_h)
            mu = 0.1
            for _ in range(200):
                mu = float((mu * pc_h[ok] / ((1 - mu) * pmh[ok] + mu * pc_h[ok])).mean())
            bits_tok = -with_cache(lpm_v, pc_v, has_v, mu) / LN2
            record(name + "+doc_cache", {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                         "heldout_bits_per_token": float((-with_cache(lpm_h, pc_h, has_h, mu) / LN2).mean()),
                                         "arm": arm, "estimator": "lin with a KN-5 expert", "induction": ind, "doc_cache_mu": mu,
                                         "top1": None, "top5": None}, bits_tok)
    TT["kn5_rows_s"] = time.time() - t0 - TT["kn_fit_s"]

    # ---------------------------------------------------------------- KN comparators on the SAME counting documents
    if comparators:
        t0 = time.time()
        comp_rows = {}
        for order in (1, 2, 3, 4, 5):
            lp2 = kn.logprob(vt, vds, order=order)
            bits_tok = -lp2
            comp_rows[f"comparator.kn{order}"] = ({"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                                   "heldout_bits_per_token": float(-kn.logprob(th, dh, order=order).mean()),
                                                   "fitted_on": "the counting documents of this phase (held-out documents removed)",
                                                   "discounts": "modified KN (Chen and Goodman)"}, bits_tok)
        lp_h = kn.logprob(th, dh, order=5)
        pos_h = np.arange(th.shape[0], dtype=np.int64)
        lo_h = doc_lo_of(dh, th.shape[0])
        pc_h, has_h = k_cache_gold(pos_h, lo_h, th, th, V)
        pm_h = 2.0 ** lp_h
        ok = has_h == 1
        mu_kn = 0.1
        for _ in range(200):
            mu_kn = float((mu_kn * pc_h[ok] / ((1 - mu_kn) * pm_h[ok] + mu_kn * pc_h[ok])).mean())
        lo_v = lo_va[:n_va]
        for order in (2, 5):
            ranks = np.zeros(n_va, np.int64)
            ranks_c = np.zeros(n_va, np.int64)
            pgc = np.zeros(n_va)
            for i in range(0, n_va, 2048):
                pos = np.arange(i, min(n_va, i + 2048))
                ctx, valid = kn.contexts(vt, vds, pos, None, order)
                P = kn.dist(ctx, valid, order)
                assert np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)
                g = vt[pos]
                ar = np.arange(V)[None, :]
                pg = P[np.arange(pos.shape[0]), g][:, None]
                ranks[pos] = (P > pg).sum(axis=1) + ((P == pg) & (ar < g[:, None])).sum(axis=1)
                if order == 5:
                    H, hs = k_cache_rows(pos, lo_v[pos], vt, V)
                    Pc = np.where(hs[:, None] == 1, (1 - mu_kn) * P + mu_kn * H, P)
                    assert np.all(np.abs(Pc.sum(axis=1) - 1.0) < 1e-9)
                    pgc_ = Pc[np.arange(pos.shape[0]), g][:, None]
                    ranks_c[pos] = (Pc > pgc_).sum(axis=1) + ((Pc == pgc_) & (ar < g[:, None])).sum(axis=1)
                    pgc[pos] = pgc_[:, 0]
            comp_rows[f"comparator.kn{order}"][0]["top1"] = float((ranks == 0).mean())
            comp_rows[f"comparator.kn{order}"][0]["top5"] = float((ranks < 5).mean())
            if order == 5:
                bits_tok = -np.log(pgc) / LN2
                pch = np.where(has_h == 1, (1 - mu_kn) * pm_h + mu_kn * pc_h, pm_h)
                comp_rows["comparator.kn5+doc_cache"] = ({"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                                          "top1": float((ranks_c == 0).mean()), "top5": float((ranks_c < 5).mean()),
                                                          "heldout_bits_per_token": float((-np.log2(pch)).mean()), "doc_cache_mu": mu_kn}, bits_tok)
        kn_all = KNModel(5, V, discount_mode="modified").fit(tok_tr, ds_tr)
        for order in (2, 5):
            bits_tok = -kn_all.logprob(vt, vds, order=order)
            comp_rows[f"comparator.kn{order}_whole_prefix"] = ({"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                                                "fitted_on": "the whole prefix (counting + held-out documents)"}, bits_tok)
        del kn_all
        bits_tok = -np.log2(p_uni[vt])
        comp_rows["comparator.unigram_addone"] = ({"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean())}, bits_tok)
        for nm, (row, bits_tok) in comp_rows.items():
            fname = f"{tag}.{nm}"
            ROWS[fname] = row
            PER_TOK[fname] = bits_tok.astype(np.float32)
            for kk, vv in row.items():
                M(f"rows.{fname}.{kk}", vv, note="in-house KN (src/tmlm/eval/baselines/ngram.py), same validation tokens and bytes")
            log(f"{fname}: {row}")
        TT["comparators_s"] = time.time() - t0
    del kn
    gpu.drop_csr("full")
    del gpu, tok_va_dev, sets
    pool.free_all_blocks()
    gc.collect()

    # best row of this phase by HELD-OUT bits (validation is never used to choose)
    cands = [k for k in ROWS if k.startswith(tag + ".") and ".comparator." not in k and ROWS[k].get("heldout_bits_per_token") is not None]
    best = min(cands, key=lambda k: ROWS[k]["heldout_bits_per_token"])
    pm("best_row_by_heldout_bits", best)
    TT["phase_wall_s"] = time.time() - tph
    peak = proc_status_gib("VmHWM")
    pm("timing_s", TT)
    pm("peak_rss_gib", peak, note="process VmHWM since the phase start")
    flush_metrics()
    save_outputs(X)
    mem_note(f"end of phase {tag}")
    return {"wall": TT["phase_wall_s"], "fwd_s": TT["prefix_forward_s"], "counts_s": TT["counts_s"], "ngram_s": TT["ngram_s"],
            "kn_s": TT["kn_fit_s"], "peak_rss_gib": peak, "key_bytes": key_bytes, "csr_bytes": csr_bytes, "n_tokens": n_tr,
            "best": best}


# ------------------------------------------------------------------ paired comparisons
def comparisons(X, tags: dict) -> None:
    """Document-block bootstrap (the causal probe's doc_boot: 60 documents, 2000 resamples, fixed seed) of paired
    differences. Sign convention: every 'gain', 'share', 'margin' and 'Delta' is positive when the second-named model
    (the richer one) needs fewer bits."""
    doc_of_val, n_val_docs = X["doc_of_val"], X["n_val_docs"]
    ratio = X["n_va"] / X["n_bytes"] if X["n_va"] == X["n_va_tok"] else None

    def g(name):
        v = PER_TOK.get(name)
        return None if v is None else v.astype(np.float64)

    def P(label, a, b, a_name, b_name, note=""):
        if a is None or b is None:
            return
        mean, ci, nd = doc_boot(a, b, doc_of_val, n_val_docs)
        M(f"paired.{label}", {"a": a_name, "b": b_name, "bits_per_token_a_minus_b": mean, "ci95_bits_per_token": ci, "docs": nd,
                              "bits_per_byte_a_minus_b": (mean * ratio) if ratio else None,
                              "ci95_bits_per_byte": [c * ratio for c in ci] if ratio else None},
          label="derived", note=("document-block bootstrap, 2000 resamples; bits per byte = bits per token x tokens / bytes of the full split. " + note).strip())

    variants = ["lin", "lin+induction", "lin+doc_cache", "lin+induction+doc_cache"]
    kvariants = ["lin", "lin+doc_cache", "lin+induction", "lin+induction+doc_cache"]
    small, young, large = tags.get("small"), tags.get("young"), tags.get("large")

    def ng_tag(t):
        return small if t == young else t

    def share_vecs(t):
        """TM share per token: n-gram row minus union row (positive = the TM clauses save bits)."""
        out = {}
        for v in variants:
            a, b = g(f"{ng_tag(t)}.ngram.{v}"), g(f"{t}.union.{v}")
            out[v] = None if a is None or b is None else (a, b, f"{ng_tag(t)}.ngram.{v}", f"{t}.union.{v}")
        for v in kvariants:
            a, b = g(f"{ng_tag(t)}.kn5+ngram.{v}"), g(f"{t}.kn5+union.{v}")
            out["kn5+" + v] = None if a is None or b is None else (a, b, f"{ng_tag(t)}.kn5+ngram.{v}", f"{t}.kn5+union.{v}")
        return out

    def ind_vecs(t):
        out = {}
        for arm in ("engine", "ngram", "union"):
            tt = ng_tag(t) if arm == "ngram" else t
            for c in ("", "+doc_cache"):
                a, b = g(f"{tt}.{arm}.lin{c}"), g(f"{tt}.{arm}.lin+induction{c}")
                out[f"{arm}{c}"] = None if a is None or b is None else (a, b, f"{tt}.{arm}.lin{c}", f"{tt}.{arm}.lin+induction{c}")
        for arm in ("ngram", "union"):
            tt = ng_tag(t) if arm == "ngram" else t
            for c in ("", "+doc_cache"):
                a, b = g(f"{tt}.kn5+{arm}.lin{c}"), g(f"{tt}.kn5+{arm}.lin+induction{c}")
                out[f"kn5+{arm}{c}"] = None if a is None or b is None else (a, b, f"{tt}.kn5+{arm}.lin{c}", f"{tt}.kn5+{arm}.lin+induction{c}")
        return out

    def margin_vecs(t):
        ct = small if t == young else t
        out = {}
        rows = {"best": None, "kn5+union.lin+induction+doc_cache": f"{t}.kn5+union.lin+induction+doc_cache",
                "kn5+ngram.lin+induction+doc_cache": f"{ng_tag(t)}.kn5+ngram.lin+induction+doc_cache",
                "union.lin+induction+doc_cache": f"{t}.union.lin+induction+doc_cache", "engine.lin": f"{t}.engine.lin",
                "engine.lin+induction+doc_cache": f"{t}.engine.lin+induction+doc_cache"}
        best = METRICS.get(f"{t}.best_row_by_heldout_bits", {}).get("value")
        rows["best"] = best
        for rn, full in rows.items():
            if full is None:
                continue
            for cn in ("kn5", "kn5+doc_cache", "kn2"):
                a, b = g(f"{ct}.comparator.{cn}"), g(full)
                out[f"{rn}_vs_{cn}"] = None if a is None or b is None else (a, b, f"{ct}.comparator.{cn}", full)
        return out

    per_tag = {}
    for t in [x for x in (small, young, large) if x is not None]:
        S, I, Mg = share_vecs(t), ind_vecs(t), margin_vecs(t)
        per_tag[t] = (S, I, Mg)
        for k, v in S.items():
            if v is not None:
                P(f"{t}.tm_share.{k}", v[0], v[1], v[2], v[3], "TM share = n-gram row minus union row")
        for k, v in I.items():
            if v is not None:
                P(f"{t}.induction_gain.{k}", v[0], v[1], v[2], v[3], "induction gain = row without minus row with induction")
        for k, v in Mg.items():
            if v is not None:
                P(f"{t}.margin.{k}", v[0], v[1], v[2], v[3], "margin = comparator minus row (positive = row below the comparator)")
    # (A) trained vs young, paired: Delta_trained - Delta_young = young union row - trained union row
    if small is not None and young is not None:
        for k in per_tag[small][0]:
            vt_, vy = per_tag[small][0][k], per_tag[young][0][k]
            if vt_ is None or vy is None:
                continue
            P(f"C0.delta_trained_minus_delta_young.{k}", vt_[0] - vt_[1], vy[0] - vy[1], f"Delta_trained ({vt_[2]} - {vt_[3]})",
              f"Delta_young ({vy[2]} - {vy[3]})", "equals the young union row minus the trained union row (n-gram rows shared)")
        for v in variants:
            a, b = g(f"{young}.engine.{v}"), g(f"{small}.engine.{v}")
            P(f"C0.young_minus_trained.engine.{v}", a, b, f"{young}.engine.{v}", f"{small}.engine.{v}", "clause arm alone: positive = trained clauses better")
        a, b = g(f"{small}.ngram.lin"), g(f"{young}.ngram.lin")
        if a is not None and b is not None:
            M("C0.check_ngram_rows_identical", float(np.abs(a - b).max()), note="the n-gram arm does not depend on the bank: max abs per-token difference between the two phases")
    # (B) scale: quantity at the large N minus the same quantity at the small N
    if small is not None and large is not None:
        for kind, j in (("tm_share", 0), ("induction_gain", 1), ("margin", 2)):
            for k in per_tag[small][j]:
                vs, vl = per_tag[small][j][k], per_tag[large][j].get(k)
                if vs is None or vl is None:
                    continue
                if kind == "margin" and k.startswith("best_"):
                    note = f"best rows: small {METRICS.get(small + '.best_row_by_heldout_bits', {}).get('value')}, large {METRICS.get(large + '.best_row_by_heldout_bits', {}).get('value')}"
                else:
                    note = ""
                P(f"scale.{kind}.{k}.large_minus_small", vl[0] - vl[1], vs[0] - vs[1], f"{kind} at {large}", f"{kind} at {small}",
                  ("positive = the quantity grows with N. " + note).strip())
    flush_metrics()


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CP_DIR / "checkpoint_epoch4.npz"))
    ap.add_argument("--n-small", type=int, default=5_000_000)
    ap.add_argument("--n-large", default="20000000,15000000,10000000", help="candidates, largest first; the first that fits is used")
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--val-max-pos", type=int, default=0, help="smoke only")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--rbits", type=int, default=17)
    ap.add_argument("--fit-every", type=int, default=3)
    ap.add_argument("--lin-iters", type=int, default=300)
    ap.add_argument("--kn5-engine", type=int, default=0)
    ap.add_argument("--young-fill", type=float, default=0.98)
    ap.add_argument("--young-max-batches", type=int, default=20000)
    ap.add_argument("--young-checkpoint", default=None, help="reuse a saved young bank instead of training one")
    ap.add_argument("--phases", default="young,small,large")
    ap.add_argument("--wall-cap-s", type=float, default=9000.0)
    ap.add_argument("--mem-budget-gib", type=float, default=52.0, help="projected process peak allowed for the large phase")
    ap.add_argument("--scratch-dir", default=os.environ.get("TMLM_SCRATCH", "/tmp"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    M("mode_smoke", bool(args.smoke))
    M("args", vars(args))

    import cupy as cp  # noqa: F401
    from tmlm.data.corpus_store import load_split
    from tmlm.data.evalsets import test_access_count
    from tmlm.engine.checkpoint import load_checkpoint
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import contiguous, split_meta
    M("test_accesses_before", test_access_count())

    cfg, st_trained, meta = load_checkpoint(args.checkpoint)
    cp_metrics = json.loads((CP_DIR / "metrics.json").read_text())
    assert meta["state_hash"] == cp_metrics["epoch4.checkpoint"]["value"]["state_hash"], "checkpoint state hash differs from the logged epoch-4 hash"
    assert cfg.mode == "causal"
    V, C = int(cfg.vocab), int(cfg.n_clauses)
    M("trained.state_hash", meta["state_hash"], note="equals epoch4.checkpoint.state_hash of results/20260923T042350Z_CP_anchor")
    M("trained.engine_config", cfg.to_dict())
    full = load_split(CORPUS, V, "train")
    train_counts_full = np.bincount(np.asarray(full.tok).astype(np.int64), minlength=V)
    val = contiguous(load_split(CORPUS, V, "validation"))
    val_meta, _ = split_meta(V, "validation")
    n_va_tok = int(val.tok.shape[0])
    n_va = n_va_tok if args.val_max_pos <= 0 else min(args.val_max_pos, n_va_tok)
    tok_va = np.asarray(val.tok).astype(np.int64)
    ds_va = np.asarray(val.doc_start).astype(np.int64)
    small = contiguous(slice_train(full, args.n_small))
    ds_small = np.asarray(small.doc_start).astype(np.int64)
    n_doc_small = ds_small.shape[0] - 1
    del small
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, 1]))
    n_ho = max(1, int(round(args.heldout_frac * n_doc_small)))
    ho_docs = np.sort(rng.choice(n_doc_small, size=n_ho, replace=False))
    doc_of_val = (np.searchsorted(ds_va, np.arange(n_va), side="right") - 1).astype(np.int64)
    X = dict(tok_va=tok_va, lo_va=doc_lo_of(ds_va, n_va_tok), ds_va=ds_va, n_va=n_va, n_va_tok=n_va_tok, n_bytes=int(val_meta["n_bytes_utf8"]),
             val_corpus=val, ds_small=ds_small, ho_docs=ho_docs, doc_of_val=doc_of_val, n_val_docs=int(ds_va.shape[0] - 1),
             cp_metrics=cp_metrics, train_counts_full=train_counts_full)
    M("split.heldout_doc_ids", ho_docs.tolist(), note="SeedSequence([20260923, 715, 1]) over the documents of the small prefix; the same documents at every N")
    M("data.validation_tokens", n_va_tok)
    M("data.validation_bytes_utf8", X["n_bytes"])
    M("data.validation_docs", X["n_val_docs"])
    M("trained.bank_summary", bank_summary(st_trained, V))
    flush_metrics()
    phases = args.phases.split(",")
    out_dir = Path(os.environ.get("TMLM_RUN_DIR") or args.scratch_dir)
    tags: dict = {}
    mb = lambda n: f"{n / 1e6:g}M"  # noqa: E731

    # ---------------------------------------------------------------- (A) young bank
    st_young = None
    if "young" in phases:
        if args.young_checkpoint:
            cfg_y, st_young, meta_y = load_checkpoint(args.young_checkpoint)
            assert cfg_y == cfg
            M("young.checkpoint", {"path": args.young_checkpoint, "state_hash": meta_y["state_hash"], "reused": True, "position": meta_y["position"]})
        else:
            full_c = contiguous(full)
            st_young, _, _ = train_young(cfg, full_c, args, out_dir)
            del full_c
            gc.collect()
        M("young.bank_summary", bank_summary(st_young, V))
        flush_metrics()
        mem_note("after the young bank")

    # ---------------------------------------------------------------- phases
    stats = {}
    if "small" in phases:
        t = f"trained{mb(args.n_small)}"
        tags["small"] = t
        stats["small"] = run_phase(t, st_trained, cfg, full, args.n_small, X, args, trained=True, comparators=True, ngram_rows="all")
        comparisons(X, tags)
        # C2: this run's trained small-prefix rows against the causal probe's logged rows (same code path, same inputs)
        if args.n_small == 5_000_000 and n_va == n_va_tok and (OLD_RUN / "rows.json").exists():
            old = json.loads((OLD_RUN / "rows.json").read_text())
            oldb = np.load(OLD_RUN / "per_token_bits.npz")
            rep = {}
            for k, row in old.items():
                nk = f"{t}.{k}"
                if nk in ROWS and row.get("bits_per_byte") is not None and ROWS[nk].get("bits_per_byte") is not None:
                    d_bpb = abs(ROWS[nk]["bits_per_byte"] - row["bits_per_byte"])
                    d_tok = float(np.abs(PER_TOK[nk].astype(np.float64) - oldb[k].astype(np.float64)).max()) if k in oldb.files else None
                    rep[k] = {"abs_diff_bpb": d_bpb, "max_abs_diff_per_token_bits": d_tok,
                              "abs_diff_top1": (abs(ROWS[nk]["top1"] - row["top1"]) if ROWS[nk].get("top1") is not None and row.get("top1") is not None else None)}
            M("C2.reproduction_of_causal_probe_rows", rep, note="this run's trained 5M rows vs results/20260924T012544Z_ckm_probe_causal_exploratory rows.json and per_token_bits.npz")
            worst = max((v["abs_diff_bpb"] for v in rep.values()), default=None)
            M("C2.max_abs_diff_bpb", worst)
            M("C2.passed", bool(rep) and worst is not None and worst < 1e-9)
            log(f"C2 reproduction: {len(rep)} rows compared, max |d bpb| {worst}")
            flush_metrics()
    if "young" in phases and st_young is not None:
        t = f"young{mb(args.n_small)}"
        tags["young"] = t
        stats["young"] = run_phase(t, st_young, cfg, full, args.n_small, X, args, trained=False, comparators=False, ngram_rows="check")
        del st_young
        gc.collect()
        comparisons(X, tags)
    if "large" in phases:
        cands = [int(x) for x in args.n_large.split(",") if x.strip()]
        chosen = None
        dec = []
        s5 = stats.get("small")
        for N in cands:
            if s5 is None:
                chosen = N
                dec.append({"N": N, "note": "no small-phase measurement, taken as is"})
                break
            sc = N / s5["n_tokens"]
            proj_wall = s5["wall"] + (s5["fwd_s"] + s5["counts_s"] + s5["ngram_s"] + s5["kn_s"]) * (sc - 1.0)
            proj_mem = s5["peak_rss_gib"] + (s5["key_bytes"] + 2.0 * s5["csr_bytes"]) * (sc - 1.0) / 2 ** 30
            left = args.wall_cap_s - (time.time() - T0)
            ok = proj_wall <= 0.9 * left and proj_mem <= args.mem_budget_gib
            dec.append({"N": N, "projected_wall_s": proj_wall, "wall_left_s": left, "projected_peak_rss_gib": proj_mem,
                        "mem_budget_gib": args.mem_budget_gib, "fits": ok})
            if ok:
                chosen = N
                break
        M("scale.choice", {"chosen_N": chosen, "decisions": dec}, label="derived",
          note="projection from the small phase: scaled parts (forward, counts, n-gram tables, KN fit) linear in N; memory = small peak + (count keys + 2 x count table) x (N/N_small - 1)")
        flush_metrics()
        log(f"scale point: {dec} -> {chosen}")
        if chosen is not None:
            t = f"trained{mb(chosen)}"
            tags["large"] = t
            stats["large"] = run_phase(t, st_trained, cfg, full, chosen, X, args, trained=True, comparators=True, ngram_rows="all")
            comparisons(X, tags)
    M("tags", tags)
    M("test_accesses_after", test_access_count())
    M("wall_s_total", time.time() - T0)
    flush_metrics()
    save_outputs(X)
    if not os.environ.get("TMLM_RUN_DIR"):
        print(json.dumps(py({k: {kk: vv for kk, vv in v.items() if kk in ("bits_per_byte", "bits_per_token", "top1", "heldout_bits_per_token")} for k, v in ROWS.items()}), indent=1))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
