"""Confirmatory rows H8.2, H8.7, H8.8 and H8.9 of docs/13 amendment A3 (lane A1). VALIDATION ONLY: the
test split is never loaded (only meta.json digests of train and validation are read for the dataset
hashes). The CVTM-core is deterministic given the data (no seed, docs/13 A3); intervals are the
document-block bootstrap of `tmlm.cvtm.protocol` (60 validation documents, 2,000 resamples, labelled
derived). The frozen CVTM-core and the KN-5 files come from the L0 cache
`results/20260924T085728Z_cvtm_cache_build/cache/` (state hash 97052edf...).

One mode per run (config `experiments/cvtm/configs/cvtm_a3_<mode>.json`, key "mode"):

- H8_2: the suffix clauses of the frozen core (float64 votes) against `KNModel.prob` at every validation
  position, and the Q16 integer form (bits per byte). CPU only.
- H8_7: CVTM-core with document window N in {0, 16, 64, 256, 1024, 4096, 16384, 65536, whole document};
  for EVERY N the EM weights are refitted on H (each N is a model variant fitted by the 11.6 protocol;
  `CVTMCore.variant(window=N)`), the suffix clauses are those of the frozen core. (a) non-increasing
  within 0.001, (b) per-token perplexity at 4096 at least 5 percent below N = 0, (c) ns per token of
  the FULL scoring pass (suffix lookups with `model.suffix_gold`, streaming document clauses with
  `doc_clauses.doc_features_stream`, mixture with `mixture.gold_components`, `bucket_ids`, `gold_prob`),
  single thread (run under scripts/cpu_slot.sh 1), one untimed warm-up pass per window, then 5 timed
  repeats with the two windows interleaved (16, 65536, 16, 65536, ...), median; CPU only.
- H8_8a: `CVTMCore.fit` on training prefixes of whole documents (about 5M, 20M, 67M tokens, and the full
  split; H chosen inside each prefix by the residue rule); validation bits per byte strictly decreasing.
  The full-split fit must reproduce the frozen core's state hash.
- H8_8b: `tmlm.cvtm.shard.fit_kn_sharded` with S = 4 on the full training split against the unsharded
  `kn_gpu.fit_kn`: every KN array and discount (`kn_gpu.compare_models`, element by element), every
  suffix-clause array (sha256 of dtype, shape and bytes of keys, fires, d, v, dq, vq, b, bq, p1), and the
  frozen core's state hash rebuilt over the sharded suffix clauses.
- H8_8c: end-to-end fit throughput, run under the EXCLUSIVE GPU lock: 5 repeats of `CVTMCore.fit` on the
  full split in one process (tokens already in host memory; the data load is timed and reported
  separately), throughput = training tokens of the split / fit wall time, median of the 5 repeats (the
  first repeat is included; its CUDA context start is part of it). Every repeat must reproduce the
  frozen state hash.
- H8_9: `CVTMCore.fit(min_counts=...)` with the F20 d estimator (prune_cont "full", prune_mass "backoff")
  for {3: 2, 4: 2, 5: 2} (count-1 types removed at orders 3 to 5) and {3: 3, 4: 3, 5: 3} (count at most 2
  removed); model bytes = `SuffixClauses.nbytes()`; the unpruned reference is the frozen core.

Every hypothesis gets a metric "verdict.<id>" with its number, threshold and interval; a miss is a
result (exit 0), a failed pipeline check makes the run exit 1.

Launch patterns (the config of each mode repeats its own):

    CUDA_VISIBLE_DEVICES= scripts/cpu_slot.sh 1 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_a3_H8_7 --config experiments/cvtm/configs/cvtm_a3_H8_7.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_a3_rows.py --config experiments/cvtm/configs/cvtm_a3_H8_7.json
    scripts/gpu_run.sh shared -- scripts/cpu_slot.sh 1 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_a3_H8_8a --config ... -- .venv-tm/bin/python -u experiments/cvtm_a3_rows.py --config ...

Smoke (not logged; scratch numbers only, every reference check is recorded but cannot fail):
`--smoke` fits a small core on a 3M-token training prefix with the numpy backend and uses the first
validation documents.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.cvtm import cache as C  # noqa: E402
from tmlm.cvtm import kn_gpu, protocol  # noqa: E402
from tmlm.cvtm.doc_clauses import doc_features, doc_features_stream  # noqa: E402
from tmlm.cvtm.mixture import D_ORDERS, bucket_ids, counts_matrix, gold_components, gold_prob, kn_gold_and_depth  # noqa: E402
from tmlm.cvtm.model import PTR_MAX, PTR_MIN, CVTMCore, suffix_gold  # noqa: E402
from tmlm.cvtm.shard import compare_suffix_clauses, fit_kn_sharded  # noqa: E402
from tmlm.cvtm.suffix import SuffixClauses  # noqa: E402
from tmlm.eval.baselines.ngram import bits_per_byte  # noqa: E402

LN2 = math.log(2.0)
GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False
WINDOWS_LABEL = "whole"


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[a3_rows {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def _plain(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def M(name: str, value, label: str = "measured", **extra) -> None:
    value = _plain(value)
    extra = _plain(extra)
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


def verdict(hid: str, met: bool, *, value, threshold: str, ci=None, **extra) -> None:
    """The pre-registered verdict of one hypothesis: MET or NOT MET (a miss is a result, not an error)."""
    M(f"verdict.{hid}", "MET" if met else "NOT MET", label="derived", number=value, threshold=threshold,
      ci95=ci, **extra)


# ------------------------------------------------------------------ memory and contention
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


def wait_for_memory(peak_gib: float, tag: str, poll_s: int = 120, max_wait_s: int = 8 * 3600) -> None:
    """The memory rule: start only when memory.current + expected peak < 0.80 memory.max (poll every
    2 minutes). The working set is logged next to it."""
    waited = 0
    if SMOKE:
        peak_gib = 2.0                         # smoke data are small; the full-data estimate does not apply
    while True:
        cur, ws, mx = cgroup()
        if cur + peak_gib * GIB < 0.80 * mx:
            log(f"memory before {tag}: current {cur / GIB:.1f} GiB (working set {ws / GIB:.1f}) + peak {peak_gib:.0f} GiB "
                f"< 0.80 x {mx / GIB:.0f} GiB (waited {waited} s)")
            M(f"memory.before.{tag}", {"current_gib": round(cur / GIB, 2), "working_set_gib": round(ws / GIB, 2),
                                       "expected_peak_gib": peak_gib, "waited_s": waited})
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag}")
        log(f"memory before {tag}: current {cur / GIB:.1f} GiB (working set {ws / GIB:.1f}) + {peak_gib:.0f} GiB "
            f">= 0.80 x max; waiting {poll_s} s")
        time.sleep(poll_s)
        waited += poll_s


class PeakSampler:
    def __init__(self) -> None:
        self.peak_ws = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_ws = max(self.peak_ws, cgroup()[1])
            except (OSError, ValueError):
                pass
            self._stop.wait(2.0)

    def stop(self) -> int:
        self._stop.set()
        self._t.join()
        return self.peak_ws


def rss_peak_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0 ** 2
    return float("nan")


def cgroup_cpu_usec() -> int:
    for line in (CG / "cpu.stat").read_text().splitlines():
        k, _, v = line.partition(" ")
        if k == "usage_usec":
            return int(v)
    return 0


class CpuWindow:
    """Wall time, own CPU time, and the CPU time of every OTHER process of the container over a timed
    window (cgroup cpu.stat usage minus this process); host load averages. Other tenants of the host
    are invisible here, so every timing is labelled contended (shared host)."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.c0 = cgroup_cpu_usec()
        self.p0 = time.process_time()
        self.l0 = os.getloadavg()

    def close(self) -> dict:
        wall = time.perf_counter() - self.t0
        cont = (cgroup_cpu_usec() - self.c0) / 1e6
        own = time.process_time() - self.p0
        return {"wall_s": wall, "own_cpu_s": own, "container_cpu_s": cont,
                "other_container_processes_cpu_cores_avg": max(0.0, cont - own) / max(wall, 1e-9),
                "loadavg_start": list(self.l0), "loadavg_end": list(os.getloadavg())}


def free_gpu() -> None:
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def drop_file_cache(path: Path) -> None:
    """posix_fadvise(DONTNEED) on a file this run has read (no data changes): the page cache of the
    large cache files would otherwise count against the memory rule."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


# ------------------------------------------------------------------ data and cores
def load_data(cfg: dict, args, need_train: bool = True):
    V = int(cfg["data"]["vocab"])
    t0 = time.time()
    M("dataset_hashes", protocol.write_dataset_hashes(("train", "validation"), protocol.CORPUS, V),
      note="sha256 entries of the stores' meta.json files (also in dataset_hashes.json)")
    tok = ds = None
    if need_train:
        tok, ds = protocol.load_train(protocol.CORPUS, V, check=True)
        tok = np.ascontiguousarray(np.asarray(tok))
    ev = protocol.load_eval_split("validation", protocol.CORPUS, V, check=True)
    if SMOKE:
        if need_train:
            tok, ds = protocol.prefix_documents(tok, ds, args.smoke_train_tokens)
            tok = np.ascontiguousarray(tok)
        kd = args.smoke_val_docs
        vds = ev.doc_start[:kd + 1]
        ev = protocol.EvalSplit(split="validation", tok=ev.tok[:int(vds[-1])], doc_start=vds, doc_bytes=ev.doc_bytes[:kd],
                                n_bytes=int(ev.doc_bytes[:kd].sum()), n_words=0, meta=ev.meta, raw_meta=ev.raw_meta)
    M("data.load_wall_s", round(time.time() - t0, 2), unit="s", note="memory-mapped stores, every array re-hashed")
    if need_train:
        M("data.train_tokens", int(ds[-1]))
        M("data.train_docs", int(ds.shape[0] - 1))
    M("data.validation_tokens", int(ev.tok.shape[0]))
    M("data.validation_docs", int(ev.n_docs))
    M("data.validation_bytes_utf8", int(ev.n_bytes), note="protocol CP byte accounting")
    M("data.validation_max_doc_tokens", int(np.diff(ev.doc_start).max()))
    return tok, ds, ev


def cache_dir(cfg: dict) -> Path:
    return ROOT / cfg["frozen"]["cache_dir"]


def frozen_core(cfg: dict, args, backend: str, tok=None, ds=None) -> CVTMCore:
    """The frozen CVTM-core of the L0 cache (state hash checked), or in smoke mode a small core fitted
    on the smoke prefix with the numpy backend."""
    if SMOKE:
        t0 = time.time()
        core = CVTMCore.fit(tok, ds, vocab=int(cfg["data"]["vocab"]), backend="numpy", expect_h=False, log=log)
        M("core.smoke_fit_wall_s", round(time.time() - t0, 1), unit="s")
        return core
    cd = cache_dir(cfg)
    t0 = time.time()
    core = C.load_core(cd, backend=backend)
    M("core.load_wall_s", round(time.time() - t0, 1), unit="s", note=f"CVTMCore.load with suffix backend {backend!r}")
    drop_file_cache(cd / C.CORE_FILE)
    check("core.state_hash_is_frozen", core.state_hash() == cfg["frozen"]["core_state_hash"],
          reference_check=True, got=core.state_hash(), want=cfg["frozen"]["core_state_hash"])
    return core


def freeze_check(cfg: dict) -> None:
    """The stage-1 library files must still carry the L0 freeze hashes."""
    d = ROOT / "src" / "tmlm" / "cvtm"
    got = {p.name: hashing.sha256_file(p) for p in sorted(d.glob("*.py"))}
    want = cfg["frozen"]["cvtm_sources_sha256"]
    same = {k: got.get(k) == v for k, v in want.items()}
    M("freeze.cvtm_sources_sha256", got, note="shard.py is the new module of lane A1; the other files are L0's frozen set")
    check("freeze.l0_files_unchanged", all(same.values()), reference_check=True,
          changed=[k for k, ok in same.items() if not ok])


# ------------------------------------------------------------------ scoring helpers
def score(core: CVTMCore, ev, lnp: np.ndarray, ks: np.ndarray, feats=None) -> dict:
    rec = core.gold(ev.tok, ev.doc_start, lnp_s=lnp, kstar=ks, feats=feats)
    return core.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes, doc_bytes=ev.doc_bytes, rec=rec)


def row(name: str, sc: dict, idx: np.ndarray, ev) -> None:
    M(f"validation.{name}.bits_per_byte", sc["bits_per_byte"], note="total negative log2-likelihood / UTF-8 bytes (protocol CP)")
    M(f"validation.{name}.bits_per_token", sc["bits_per_token"])
    M(f"validation.{name}.token_perplexity", 2.0 ** sc["bits_per_token"], label="derived")
    b = protocol.bootstrap_bpb(sc["doc_bits"], ev.doc_bytes, idx)
    M(f"validation.{name}.bits_per_byte_ci95", b["ci95"], label="derived",
      note=f"document-block bootstrap: {b['n_docs']} validation documents, {b['n_boot']} resamples, seed {list(protocol.BOOT_SEED)}")


def paired(name: str, a: dict, b: dict, idx: np.ndarray, ev) -> dict:
    r = protocol.bootstrap_bpb(a["doc_bits"], ev.doc_bytes, idx, b["doc_bits"])
    M(f"validation.{name}.bits_per_byte", r["point"], label="derived", ci95=r["ci95"],
      note="paired document-block bootstrap (validation documents, 2,000 resamples)")
    return r


def em_checks(tag: str, trace: dict) -> None:
    dll = np.diff(np.asarray(trace["pooled_ll"]))
    dob = np.diff(np.asarray(trace["bucket_map_objective"]))
    check(f"em.{tag}.pooled_ll_non_decreasing", bool(dll.min() >= -1e-12), min_step=float(dll.min()))
    check(f"em.{tag}.map_objective_non_decreasing", bool(dob.min() >= -1e-12), min_step=float(dob.min()))
    M(f"em.{tag}.pooled_weights", trace["pooled_weights"])
    M(f"em.{tag}.buckets_with_fit_rows", int(np.count_nonzero(trace["rows_per_bucket"])),
      note=f"of {len(trace['rows_per_bucket'])}")


# ------------------------------------------------------------------ H8.2
def mode_h8_2(cfg: dict, args) -> None:
    tok, ds, ev = load_data(cfg, args, need_train=SMOKE)
    wait_for_memory(float(cfg["compute"]["peak_gib"]), "core_load")
    core = frozen_core(cfg, args, backend="numpy", tok=tok, ds=ds)
    kn = core.kn
    sc = core.suffix
    t0 = time.time()
    lnp, ks = suffix_gold(sc, ev.tok, ev.doc_start)
    M("time.vote_form_s", round(time.time() - t0, 2), unit="s")
    t0 = time.time()
    p_ref, k_ref = kn_gold_and_depth(kn, ev.tok, ev.doc_start)
    M("time.KNModel_prob_s", round(time.time() - t0, 2), unit="s")
    diff = np.abs(lnp - np.log(p_ref))
    dmax = float(diff.max())
    M("h8_2.max_abs_ln_diff", dmax, note="max over every validation position of |ln P_clause - ln KNModel.prob| (natural log)")
    M("h8_2.n_positions", int(diff.shape[0]))
    M("h8_2.n_positions_exactly_equal", int(np.count_nonzero(diff == 0.0)))
    check("kstar_equals_kn_gold_and_depth", bool(np.array_equal(ks.astype(np.int64), k_ref)))
    # the stream forms of the suffix clauses (the receipts' form)
    lnp_stream = sc.logprob_stream(ev.tok, ev.doc_start)
    check("logprob_stream_equals_suffix_gold", bool(np.array_equal(lnp_stream, lnp)))
    lq = sc.logprob_stream(ev.tok, ev.doc_start, quantized=True)
    bpb_f = float(-lnp.sum() / LN2 / ev.n_bytes)
    bpb_q = float(-(lq.astype(np.float64) / 65536.0).sum() / LN2 / ev.n_bytes)
    bpb_knmodel = bits_per_byte(kn.logprob(ev.tok, ev.doc_start), ev.n_bytes)
    M("h8_2.float_vote_validation_bits_per_byte", bpb_f)
    M("h8_2.q16_validation_bits_per_byte", bpb_q, note="int32 votes in units of 2^-16 nats summed in int64 per position, / 65536 / ln 2 / bytes")
    M("h8_2.knmodel_validation_bits_per_byte", bpb_knmodel, note="KNModel.logprob path (the M2 reference)")
    dq = bpb_q - bpb_f
    M("h8_2.q16_minus_float_bits_per_byte", dq, label="derived")
    check("knmodel_path_equals_M2", bpb_knmodel == float(cfg["reference"]["kn5_validation_bpb"]), reference_check=True,
          got=bpb_knmodel, want=cfg["reference"]["kn5_validation_bpb"])
    # every integer sum lies within its documented rounding bound of the float64 sum
    ctx_all_ok = True
    worst = 0.0
    for lo in range(0, ev.tok.shape[0], 1 << 16):
        pos = np.arange(lo, min(lo + (1 << 16), ev.tok.shape[0]))
        ctx, valid = sc.contexts(ev.tok, ev.doc_start, pos)
        bound = sc.rounding_bound(ctx, ev.tok[pos], valid)
        err = np.abs(lq[pos].astype(np.float64) / 65536.0 - lnp[pos])
        worst = max(worst, float((err - bound).max()))
        ctx_all_ok &= bool(np.all(err <= bound + 1e-12))
    check("q16_within_rounding_bound_every_position", ctx_all_ok, max_excess=worst)
    thr = cfg["thresholds"]
    met_a = dmax <= float(thr["max_abs_ln_diff"])
    met_b = abs(dq) <= float(thr["q16_abs_bpb_change"])
    verdict("H8.2", met_a and met_b, value={"max_abs_ln_diff": dmax, "q16_minus_float_bits_per_byte": dq},
            threshold=f"max |ln P_clause - ln P_KN| <= {thr['max_abs_ln_diff']} AND |Q16 bpb - float bpb| <= {thr['q16_abs_bpb_change']}",
            parts={"max_abs_ln_diff": "MET" if met_a else "NOT MET", "q16": "MET" if met_b else "NOT MET"},
            note="deterministic quantities over every validation position (no interval)")


# ------------------------------------------------------------------ H8.7
def full_scoring_pass(core: CVTMCore, tok: np.ndarray, ds: np.ndarray, window) -> tuple[np.ndarray, dict]:
    """The full per-token scoring pass of the CVTM-core with STREAMING document clauses:
    suffix lookups, document clauses, mixture. Returns P(gold) and the stage wall times (ns)."""
    t0 = time.perf_counter_ns()
    lnp, ks = suffix_gold(core.suffix, tok, ds)
    t1 = time.perf_counter_ns()
    feats = doc_features_stream(tok, ds, D_ORDERS, window, PTR_MIN, PTR_MAX)
    t2 = time.perf_counter_ns()
    tok64 = np.asarray(tok).astype(np.int64)
    p_comp, act = gold_components(np.exp(lnp), feats, tok64)
    bucket, _ = bucket_ids(ks.astype(np.int64), counts_matrix(feats), feats.ptr_len, core.components, PTR_MIN)
    pg = gold_prob(p_comp, act, bucket, core.weights, core.components)
    t3 = time.perf_counter_ns()
    return pg, {"suffix_ns": t1 - t0, "doc_clauses_ns": t2 - t1, "mixture_ns": t3 - t2, "total_ns": t3 - t0,
                "n_fallback": int(feats.n_fallback)}


def mode_h8_7(cfg: dict, args) -> None:
    tok, ds, ev = load_data(cfg, args, need_train=SMOKE)
    wait_for_memory(float(cfg["compute"]["peak_gib"]), "core_load")
    core = frozen_core(cfg, args, backend="numpy", tok=tok, ds=ds)
    idx = protocol.bootstrap_index(ev.n_docs)
    t0 = time.time()
    lnp, ks = suffix_gold(core.suffix, ev.tok, ev.doc_start)
    M("time.suffix_gold_validation_s", round(time.time() - t0, 2), unit="s")
    if not SMOKE:
        va = C.load_arrays(cache_dir(cfg), "validation", ["p_kn_all", "kstar_all", "core_p_gold"])
        check("suffix_gold_equals_cache", bool(np.array_equal(np.exp(lnp), va["p_kn_all"])
                                               and np.array_equal(ks, va["kstar_all"])), reference_check=True)
    windows = [None if w == WINDOWS_LABEL else int(w) for w in cfg["params"]["windows"]]
    res: dict = {}
    variants: dict = {}
    doc_tokens = np.diff(ev.doc_start).astype(np.int64)
    for w in windows:
        tag = WINDOWS_LABEL if w is None else str(w)
        t0 = time.time()
        var = core.variant(window=w)          # EM refitted on H for this window (docs/11 11.4 and 11.6)
        M(f"window.{tag}.em_refit_s", round(time.time() - t0, 1), unit="s")
        em_checks(f"window_{tag}", var.fit_info["em_trace"])
        if w is None:
            check("whole_document_refit_equals_frozen_weights", bool(np.array_equal(var.weights, core.weights)))
        t0 = time.time()
        feats = doc_features(ev.tok, ev.doc_start, orders=D_ORDERS, window=w, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
        M(f"window.{tag}.doc_features_batch_s", round(time.time() - t0, 2), unit="s")
        fs = doc_features_stream(ev.tok, ev.doc_start, D_ORDERS, w, PTR_MIN, PTR_MAX)
        check(f"window_{tag}.stream_equals_batch", bool(fs.same_values(feats)))
        sc = score(var, ev, lnp, ks, feats)
        row(f"window_{tag}", sc, idx, ev)
        M(f"validation.window_{tag}.fire_rates", {f"D{m}": float((feats.count(m) >= 1).mean()) for m in D_ORDERS}
          | {"P": float((feats.ptr_len >= PTR_MIN).mean())})
        res[tag] = sc
        if w is None and not SMOKE:
            check("whole_document_equals_frozen_core_value", sc["bits_per_byte"] == float(cfg["reference"]["cvtm_core_validation_bpb"]),
                  reference_check=True, got=sc["bits_per_byte"], want=cfg["reference"]["cvtm_core_validation_bpb"])
            check("whole_document_p_gold_equals_cache", bool(np.array_equal(sc["record"].p_gold, va["core_p_gold"])),
                  reference_check=True)
        if w == 0:
            check("window_0_equals_kn5", abs(sc["bits_per_byte"] - float(-lnp.sum() / LN2 / ev.n_bytes)) <= 1e-12,
                  got=sc["bits_per_byte"], kn5=float(-lnp.sum() / LN2 / ev.n_bytes))
        if w in cfg["params"]["timing_windows"]:
            variants[w] = var
        del sc["record"]
        gc.collect()
    tags = [WINDOWS_LABEL if w is None else str(w) for w in windows]
    bpb = [res[t]["bits_per_byte"] for t in tags]
    steps = [bpb[i + 1] - bpb[i] for i in range(len(bpb) - 1)]
    for i in range(len(tags) - 1):
        r = paired(f"window_{tags[i + 1]}.minus_window_{tags[i]}", res[tags[i + 1]], res[tags[i]], idx, ev)
        del r
    thr = cfg["thresholds"]
    met_a = max(steps) <= float(thr["a_max_increase"])
    verdict("H8.7a", met_a, value={"bits_per_byte": dict(zip(tags, bpb)), "max_step_increase": max(steps),
                                   "steps": dict(zip([f"{tags[i]}->{tags[i + 1]}" for i in range(len(steps))], steps))},
            threshold=f"bpb(N_next) - bpb(N) <= {thr['a_max_increase']} for every consecutive pair of the ordered windows",
            note="per-window bootstrap intervals and paired differences are logged under validation.*")
    # (b) per-token perplexity at 4096 vs 0: ratio = 2^(bits per token difference); paired bootstrap over documents
    a, b = res[str(thr["b_window"])], res["0"]
    ratio = 2.0 ** (a["bits_per_token"] - b["bits_per_token"])
    rb = protocol.bootstrap_bpb(a["doc_bits"], doc_tokens, idx, b["doc_bits"])
    ci = [2.0 ** rb["ci95"][0], 2.0 ** rb["ci95"][1]]
    met_b = ratio <= 1.0 - float(thr["b_min_reduction"])
    verdict("H8.7b", met_b, value={"ppl_ratio": ratio, "reduction": 1.0 - ratio,
                                   "ppl_N": 2.0 ** a["bits_per_token"], "ppl_0": 2.0 ** b["bits_per_token"]},
            threshold=f"per-token perplexity at N = {thr['b_window']} <= {1.0 - float(thr['b_min_reduction'])} x per-token perplexity at N = 0",
            ci=ci, ci_note="paired document-block bootstrap of the perplexity ratio (token-weighted), derived")
    # (c) timing of the full scoring pass
    tw = [int(w) for w in cfg["params"]["timing_windows"]]
    reps = int(cfg["params"]["timing_repeats"])
    pg_ref = {}
    for w in tw:
        feats = doc_features(ev.tok, ev.doc_start, orders=D_ORDERS, window=w, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
        pg_ref[w] = score(variants[w], ev, lnp, ks, feats)["record"].p_gold
        del feats
    del lnp, ks
    gc.collect()
    for w in tw:                                   # untimed warm-up (numba compilation, caches)
        pg, _ = full_scoring_pass(variants[w], ev.tok, ev.doc_start, w)
        check(f"timing.window_{w}.full_pass_equals_batch_path", bool(np.array_equal(pg, pg_ref[w])))
    per: dict = {w: [] for w in tw}
    cw = CpuWindow()
    for r in range(reps):
        for w in tw:
            pg, t = full_scoring_pass(variants[w], ev.tok, ev.doc_start, w)
            per[w].append(t)
            if not np.array_equal(pg, pg_ref[w]):
                check(f"timing.window_{w}.repeat_{r}_equals_batch_path", False)
    cpu = cw.close()
    n = int(ev.tok.shape[0])
    med = {}
    for w in tw:
        tot = [x["total_ns"] / n for x in per[w]]
        med[w] = float(np.median(tot))
        M(f"timing.window_{w}.ns_per_token", med[w], label="measured",
          per_repeat=tot, min=min(tot), max=max(tot),
          stages_median_ns_per_token={s: float(np.median([x[s] / n for x in per[w]])) for s in ("suffix_ns", "doc_clauses_ns", "mixture_ns")},
          contention="contended (shared host)",
          note="full scoring pass over every validation position: suffix lookups + streaming document clauses + mixture; "
               "single thread under cpu_slot.sh 1; median of the timed repeats after one untimed warm-up")
    M("timing.cpu_window", cpu, note="CPU of other processes of this container during the timed repeats; other host tenants are not visible")
    ratio_c = med[tw[1]] / med[tw[0]]
    doc_ratio = (float(np.median([x["doc_clauses_ns"] for x in per[tw[1]]])) /
                 float(np.median([x["doc_clauses_ns"] for x in per[tw[0]]])))
    M("timing.doc_clause_stage_ratio", doc_ratio, label="derived",
      note=f"streaming document-clause stage alone, N = {tw[1]} over N = {tw[0]} (not the registered quantity)")
    verdict("H8.7c", ratio_c <= float(thr["c_max_ratio"]),
            value={"ratio": ratio_c, f"ns_per_token_N{tw[0]}": med[tw[0]], f"ns_per_token_N{tw[1]}": med[tw[1]]},
            threshold=f"ns per token at N = {tw[1]} <= {thr['c_max_ratio']} x ns per token at N = {tw[0]}",
            contention="contended (shared host)",
            note=f"the longest validation document has {int(np.diff(ev.doc_start).max())} tokens, so N = {tw[1]} is the whole "
                 "document on this split")


# ------------------------------------------------------------------ H8.8 (a)
def mode_h8_8a(cfg: dict, args) -> None:
    tok, ds, ev = load_data(cfg, args, need_train=True)
    idx = protocol.bootstrap_index(ev.n_docs)
    backend = "numpy" if SMOKE else cfg["compute"]["kn_backend"]
    pool = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    prefixes = cfg["params"]["prefix_tokens"] if not SMOKE else args.smoke_prefixes
    res: dict = {}
    tags = []
    for P in prefixes:
        P = None if P in (None, "full") else int(P)
        tag = "full" if P is None else f"{P // 1_000_000}M"
        tags.append(tag)
        gc.collect()
        free_gpu()
        wait_for_memory(float(cfg["compute"]["peak_gib_full"]) if P is None else float(cfg["compute"]["peak_gib_prefix"]),
                        f"fit_{tag}")
        cw = CpuWindow()
        core = CVTMCore.fit(tok, ds, vocab=int(cfg["data"]["vocab"]), prefix_tokens=P, backend=backend,
                            max_device_bytes=pool, log=log, expect_h=None if not SMOKE else False)
        cpu = cw.close()
        fi = {k: v for k, v in core.fit_info.items() if k != "em_trace"}
        M(f"prefix_{tag}.fit_info", fi)
        M(f"prefix_{tag}.fit_wall_s", round(cpu["wall_s"], 2), unit="s", contention="shared GPU lock, shared host",
          cpu=cpu)
        M(f"prefix_{tag}.train_tokens", int(fi["train_tokens"]))
        M(f"prefix_{tag}.train_docs", int(fi["train_docs"]))
        M(f"prefix_{tag}.H", {"docs": fi["H_docs"], "tokens": fi["H_tokens"], "doc_index_sha256": fi["H_doc_index_sha256"]})
        M(f"prefix_{tag}.kn_n_types", [int(x) for x in core.kn.n_types])
        M(f"prefix_{tag}.suffix_nbytes", int(core.suffix.nbytes()))
        em_checks(f"prefix_{tag}", core.fit_info["em_trace"])
        if P is None and not SMOKE:
            check("full_split_fit_reproduces_frozen_state_hash", core.state_hash() == cfg["frozen"]["core_state_hash"],
                  reference_check=True, got=core.state_hash())
        t0 = time.time()
        lnp, ks = suffix_gold(core.suffix, ev.tok, ev.doc_start)
        feats = doc_features(ev.tok, ev.doc_start, orders=D_ORDERS, window=core.window, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
        sc = score(core, ev, lnp, ks, feats)
        del sc["record"]
        row(f"prefix_{tag}.cvtm_core", sc, idx, ev)
        kn_only = core.variant("S")
        sk = score(kn_only, ev, lnp, ks, feats)
        del sk["record"]
        row(f"prefix_{tag}.kn5", sk, idx, ev)
        paired(f"prefix_{tag}.cvtm_core.minus_kn5", sc, sk, idx, ev)
        M(f"prefix_{tag}.validation_scoring_s", round(time.time() - t0, 1), unit="s")
        if P is None and not SMOKE:
            check("full_split_value_equals_frozen", sc["bits_per_byte"] == float(cfg["reference"]["cvtm_core_validation_bpb"]),
                  reference_check=True, got=sc["bits_per_byte"])
        res[tag] = sc
        del core, kn_only, lnp, ks, feats
        gc.collect()
        free_gpu()
    bpb = [res[t]["bits_per_byte"] for t in tags]
    for i in range(len(tags) - 1):
        paired(f"prefix_{tags[i + 1]}.minus_prefix_{tags[i]}.cvtm_core", res[tags[i + 1]], res[tags[i]], idx, ev)
    strictly = all(bpb[i + 1] < bpb[i] for i in range(len(bpb) - 1))
    verdict("H8.8a", strictly, value={"bits_per_byte": dict(zip(tags, bpb)),
                                      "steps": [bpb[i + 1] - bpb[i] for i in range(len(bpb) - 1)]},
            threshold="validation bits per byte strictly decreasing over the prefixes 5M, 20M, 67M, full",
            note="paired bootstrap intervals of every step are logged under validation.prefix_*")


# ------------------------------------------------------------------ H8.8 (b)
def suffix_digest(sc: SuffixClauses) -> dict:
    """sha256 (dtype, shape, bytes) of every array compared by `shard.compare_suffix_clauses`."""
    out = {"state_hash": sc.state_hash(), "order": sc.order, "vocab": sc.vocab}
    for f in ("keys", "fires", "d", "v", "dq", "vq"):
        for j in range(1 if f == "keys" else 2, sc.order + 1):
            out[f"{f}[{j}]"] = hashing.sha256_array(np.asarray(getattr(sc, f)[j]))
    for f in ("b", "bq", "p1"):
        out[f] = hashing.sha256_array(np.asarray(getattr(sc, f)))
    return out


def mode_h8_8b(cfg: dict, args) -> None:
    tok, ds, ev = load_data(cfg, args, need_train=True)
    del ev
    V = int(cfg["data"]["vocab"])
    S = int(cfg["params"]["shards"])
    backend = "numpy" if SMOKE else cfg["compute"]["kn_backend"]
    sc_backend = "numpy" if backend == "numpy" else "auto"
    pool = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    # ---- unsharded reference
    wait_for_memory(float(cfg["compute"]["peak_gib"]), "unsharded_fit")
    cw = CpuWindow()
    kn_u = kn_gpu.fit_kn(tok, ds, order=5, vocab=V, backend=backend, max_device_bytes=pool)
    M("unsharded.fit_wall_s", round(cw.close()["wall_s"], 2), unit="s", contention="shared GPU lock, shared host")
    M("unsharded.fit_info", kn_u.fit_info)
    h_u = kn_gpu.model_hashes(kn_u)
    if not SMOKE:
        with np.load(cache_dir(cfg) / C.KN_VARIANTS["all"], allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        stored = dict(meta["kn"]["hashes"])
        stored_sc = meta["kn"]["scalars"]
        check("unsharded_equals_cached_kn5_all", {k: v for k, v in h_u.items() if k != "scalars"} == stored
              and h_u["scalars"] == stored_sc, reference_check=True)
    t0 = time.time()
    sc_u = SuffixClauses.from_kn(kn_u, backend=sc_backend)
    M("unsharded.suffix_build_s", round(time.time() - t0, 1), unit="s")
    dig_u = suffix_digest(sc_u)
    del sc_u
    gc.collect()
    free_gpu()
    # ---- sharded
    cw = CpuWindow()
    kn_s = fit_kn_sharded(tok, ds, order=5, vocab=V, shards=S, backend=backend, max_device_bytes=pool)
    M("sharded.fit_wall_s", round(cw.close()["wall_s"], 2), unit="s", contention="shared GPU lock, shared host")
    M("sharded.fit_info", kn_s.fit_info)
    t0 = time.time()
    diffs = kn_gpu.compare_models(kn_s, kn_u)
    M("compare.kn_models_s", round(time.time() - t0, 1), unit="s")
    M("compare.kn_model_differences", diffs[:50], note="kn_gpu.compare_models: every array element by element, dtypes, "
                                                        "shapes, n_types, n_train_tokens, discounts as float.hex")
    check("sharded_kn_equals_unsharded_elementwise", diffs == [], n_diffs=len(diffs))
    check("sharded_kn_hashes_equal_unsharded", kn_gpu.model_hashes(kn_s) == h_u)
    del kn_u
    gc.collect()
    t0 = time.time()
    sc_s = SuffixClauses.from_kn(kn_s, backend=sc_backend)
    M("sharded.suffix_build_s", round(time.time() - t0, 1), unit="s")
    dig_s = suffix_digest(sc_s)
    sdiff = sorted(k for k in set(dig_u) | set(dig_s) if dig_u.get(k) != dig_s.get(k))
    M("compare.suffix_digest_differences", sdiff, note="sha256 of dtype, shape and bytes of keys, fires, d, v, dq, vq per "
                                                      "level and b, bq, p1 (float64 arrays compared by their bytes)")
    M("compare.suffix_state_hash", {"unsharded": dig_u["state_hash"], "sharded": dig_s["state_hash"]})
    check("sharded_suffix_votes_equal_unsharded", sdiff == [], differences=sdiff)
    frozen_ok = None
    if not SMOKE:
        with np.load(cache_dir(cfg) / C.CORE_FILE, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
            weights = np.array(z["weights"])
        cfgc = dict(meta["config"])
        core_s = CVTMCore(suffix=sc_s, weights=weights, config=cfgc, fit_inputs=None, kn=None)
        frozen_ok = core_s.state_hash() == cfg["frozen"]["core_state_hash"]
        M("frozen_core_state_hash_over_sharded_suffix", core_s.state_hash())
        check("frozen_core_state_hash_over_sharded_suffix", frozen_ok, reference_check=True)
        del core_s
    M("sharded.balance", kn_s.fit_info["balance"], note="positions and types per shard and level")
    met = diffs == [] and sdiff == [] and (frozen_ok is not False)
    verdict("H8.8b", met, value={"kn_array_differences": len(diffs), "suffix_array_differences": len(sdiff),
                                 "shards": S, "frozen_state_hash_reproduced": frozen_ok},
            threshold="bit-identical counts, statistics, discounts and suffix votes, S = 4 shards vs unsharded fit_kn",
            note="full training split; KN arrays compared element by element, suffix arrays by sha256 of their bytes")


# ------------------------------------------------------------------ H8.8 (c)
def mode_h8_8c(cfg: dict, args) -> None:
    t0 = time.time()
    tok, ds, ev = load_data(cfg, args, need_train=True)
    load_s = time.time() - t0
    del ev
    backend = "numpy" if SMOKE else cfg["compute"]["kn_backend"]
    pool = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    n_tok = int(ds[-1])
    reps = int(cfg["params"]["repeats"]) if not SMOKE else 2
    lock = os.environ.get("TMLM_GPU_LOCK_MODE")
    M("gpu_lock_mode", lock)
    if not SMOKE:
        check("exclusive_gpu_lock_held", lock == "exclusive", got=lock)
    walls, hashes = [], []
    for r in range(reps):
        gc.collect()
        free_gpu()
        wait_for_memory(float(cfg["compute"]["peak_gib"]), f"fit_repeat_{r + 1}")
        cw = CpuWindow()
        core = CVTMCore.fit(tok, ds, vocab=int(cfg["data"]["vocab"]), backend=backend, max_device_bytes=pool,
                            expect_h=None if not SMOKE else False)
        cpu = cw.close()
        walls.append(cpu["wall_s"])
        hashes.append(core.state_hash())
        fi = {k: v for k, v in core.fit_info.items() if k != "em_trace"}
        M(f"repeat_{r + 1}", {"fit_wall_s": cpu["wall_s"], "tokens_per_s": n_tok / cpu["wall_s"], "phases_s": fi["phases_s"],
                              "state_hash": hashes[-1], "cpu": cpu},
          contention="exclusive GPU lock; host shared (other container processes and host tenants may run)")
        del core
        gc.collect()
        free_gpu()
    if not SMOKE:
        check("every_repeat_reproduces_frozen_state_hash", all(h == cfg["frozen"]["core_state_hash"] for h in hashes),
              reference_check=True, hashes=hashes)
    else:
        check("every_repeat_same_state_hash", len(set(hashes)) == 1)
    tps = [n_tok / w for w in walls]
    med = float(np.median(tps))
    M("throughput.tokens_per_s_median", med, per_repeat=tps, min=min(tps), max=max(tps),
      note="training tokens of the split / wall time of CVTMCore.fit (tokens already in host memory), median of the repeats")
    M("throughput.fit_wall_s_median", float(np.median(walls)), unit="s", per_repeat=walls)
    M("throughput.data_load_s", load_s, unit="s", note="load and re-hash of the train and validation stores, once")
    M("throughput.tokens_per_s_median_including_load", n_tok / (float(np.median(walls)) + load_s), label="derived")
    thr = float(cfg["thresholds"]["min_tokens_per_s"])
    verdict("H8.8c", med >= thr, value={"tokens_per_s_median": med, "min": min(tps), "max": max(tps), "repeats": reps,
                                        "train_tokens": n_tok},
            threshold=f"median end-to-end fit throughput >= {thr:.0f} training tokens per second",
            contention="exclusive GPU lock; host shared")


# ------------------------------------------------------------------ H8.9
def mode_h8_9(cfg: dict, args) -> None:
    tok, ds, ev = load_data(cfg, args, need_train=True)
    idx = protocol.bootstrap_index(ev.n_docs)
    backend = "numpy" if SMOKE else cfg["compute"]["kn_backend"]
    sc_backend = "numpy" if backend == "numpy" else "auto"
    pool = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    # ---- unpruned reference: the frozen core
    wait_for_memory(float(cfg["compute"]["peak_gib"]), "unpruned_reference")
    core = frozen_core(cfg, args, backend=sc_backend, tok=tok, ds=ds)
    ref_bytes = int(core.suffix.nbytes())
    M("unpruned.suffix_nbytes", ref_bytes, breakdown=core.suffix.nbytes_breakdown())
    M("unpruned.kn_model_nbytes", kn_gpu.model_nbytes(core.kn))
    M("unpruned.kn_n_types", [int(x) for x in core.kn.n_types])
    lnp, ks = suffix_gold(core.suffix, ev.tok, ev.doc_start)
    feats = doc_features(ev.tok, ev.doc_start, orders=D_ORDERS, window=None, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
    ref = score(core, ev, lnp, ks, feats)
    del ref["record"]
    row("unpruned.cvtm_core", ref, idx, ev)
    if not SMOKE:
        check("unpruned_equals_frozen_value", ref["bits_per_byte"] == float(cfg["reference"]["cvtm_core_validation_bpb"]),
              reference_check=True, got=ref["bits_per_byte"])
    del core, lnp, ks
    gc.collect()
    free_gpu()
    out = {}
    for spec in cfg["params"]["variants"]:
        name = spec["name"]
        mc = {int(k): int(v) for k, v in spec["min_counts"].items()}
        wait_for_memory(float(cfg["compute"]["peak_gib"]), f"fit_{name}")
        cw = CpuWindow()
        core = CVTMCore.fit(tok, ds, vocab=int(cfg["data"]["vocab"]), min_counts=mc, backend=backend, max_device_bytes=pool,
                            log=log, expect_h=None if not SMOKE else False)
        cpu = cw.close()
        fi = {k: v for k, v in core.fit_info.items() if k != "em_trace"}
        M(f"{name}.fit_wall_s", round(cpu["wall_s"], 1), unit="s", contention="shared GPU lock, shared host")
        M(f"{name}.pruning", {"noH": fi["kn_noH.fit_info"]["pruning"], "all": fi["kn_all.fit_info"]["pruning"]})
        em_checks(name, core.fit_info["em_trace"])
        nb = int(core.suffix.nbytes())
        M(f"{name}.suffix_nbytes", nb, breakdown=core.suffix.nbytes_breakdown())
        M(f"{name}.kn_model_nbytes", kn_gpu.model_nbytes(core.kn))
        M(f"{name}.kn_n_types", [int(x) for x in core.kn.n_types])
        frac = nb / ref_bytes
        M(f"{name}.bytes_fraction_of_unpruned", frac, label="derived")
        lnp, ks = suffix_gold(core.suffix, ev.tok, ev.doc_start)
        sc = score(core, ev, lnp, ks, feats)
        del sc["record"]
        row(f"{name}.cvtm_core", sc, idx, ev)
        loss = paired(f"{name}.cvtm_core.minus_unpruned", sc, ref, idx, ev)
        sk = score(core.variant("S"), ev, lnp, ks, feats)
        del sk["record"]
        row(f"{name}.kn5", sk, idx, ev)
        out[name] = {"min_counts": spec["min_counts"], "bytes_fraction": frac, "suffix_nbytes": nb,
                     "bits_per_byte": sc["bits_per_byte"], "loss_bits_per_byte": loss["point"], "loss_ci95": loss["ci95"]}
        del core, lnp, ks, sc, sk
        gc.collect()
        free_gpu()
    thr = cfg["thresholds"]
    qual = [n for n, o in out.items() if o["bytes_fraction"] <= float(thr["max_bytes_fraction"])
            and o["loss_bits_per_byte"] <= float(thr["max_loss_bpb"])]
    verdict("H8.9", bool(qual), value={"unpruned_suffix_nbytes": ref_bytes, "unpruned_bits_per_byte": ref["bits_per_byte"],
                                       "variants": out, "qualifying": qual},
            threshold=f"some variant with bytes <= {thr['max_bytes_fraction']} x unpruned AND loss <= {thr['max_loss_bpb']} bits per byte",
            note="loss = variant minus unpruned CVTM-core validation bits per byte; intervals are paired document-block bootstraps")


# ------------------------------------------------------------------ main
MODES = {"H8_2": mode_h8_2, "H8_7": mode_h8_7, "H8_8a": mode_h8_8a, "H8_8b": mode_h8_8b, "H8_8c": mode_h8_8c,
         "H8_9": mode_h8_9}


def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train-tokens", type=int, default=3_000_000)
    ap.add_argument("--smoke-val-docs", type=int, default=4)
    ap.add_argument("--smoke-prefixes", type=int, nargs="*", default=[1_000_000, 2_000_000])
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    if SMOKE and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py")
    cfg = json.loads(Path(args.config).read_text())
    mode = cfg["mode"]
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}")
    sampler = PeakSampler()
    M("status", cfg["status"], label="unverified")
    M("config.mode", mode)
    freeze_check(cfg)
    MODES[mode](cfg, args)
    M("peak_rss_gib", round(rss_peak_gib(), 2), note="VmHWM of the driver process")
    M("peak_cgroup_working_set_gib", round(sampler.stop() / GIB, 2), note="all processes of the container, sampled every 2 s")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    M("checks_failed", list(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
