"""CVTM language-model driver (docs/11 section 11; docs/13 amendments A3 and A4). Mode "cache_build"
(lane L0, FREEZE AND CACHE): fit the frozen CVTM-core with `tmlm.cvtm.model.CVTMCore.fit` (docs/11
11.6), fit the KN-5 variants the other lanes need, and write the cache of `tmlm.cvtm.cache` (layout,
dtypes and meaning of every array are documented there). VALIDATION ONLY: the test split is never
loaded (only the digests of meta.json files are read for dataset hashes, and not for the test split).

Steps (config `experiments/cvtm/configs/cvtm_cache_build.json`):
1. Data: train and validation of WikiText-103 at V 8000 (arrays re-hashed against meta.json);
   dataset hashes as `results/20260923T042350Z_CP_anchor/` (tokens and text, plus document starts and
   per-document bytes). H = residue 0 (asserted: 304 documents, 1,397,702 tokens), H' = residues 0, 1, 2.
2. Document clauses of every H' and validation position (batch, cross-checked against streaming).
3. `CVTMCore.fit` on the full split (GPU KN fits under the shared lock, CuPy pool capped at
   `gpu_pool_limit_gib`). Its callbacks: after the KN without H, save it and read P_KN(gold) and k* on H'
   and validation; after the EM, fit the KN without H', save it, read P_KN(gold) and k* on H' and
   validation, and compute the H' core arrays with the frozen weights over that base (docs/13 A4: the
   T2 inputs use a KN-5 fitted without H') including both top-64 lists; after the KN on all documents,
   save it and read P_KN(gold) and k* on H' and validation. Every vote-form P_KN is checked against
   `KNModel.prob` (max |ln difference|) and k* against `mixture.kn_gold_and_depth`.
4. Validation rows from the frozen implementation: KN-5 ("S"), KN-5 + D0 (the H8.4 comparator "S+D0")
   and the CVTM-core, each with the EM of docs/11 11.4 on H; document-bootstrap 95 percent intervals
   (60 documents, 2,000 resamples, labelled derived). Pipeline checks: the KNModel path gives the M2
   KN-5 number 1.3501314033898424 exactly; the vote-form KN-5 row within 1e-9 of it; the CVTM-core within
   1e-4 of the exploratory 1.2408831796986637. H8.2 quantities (max |ln P_vote - ln P_KN| over validation,
   Q16 bits per byte) are logged as checks.
5. Top-64 lists of validation (frozen core, KN on all) with row-sum and gold-entry checks; causal top-1.
6. Save the frozen core, reload it (state hash and bit-identical validation log2 P), reload every KN
   file (array hashes), write cache_meta.json with the sha256 of every file, and log the sha256 of every
   src/tmlm/cvtm/*.py (the freeze).

Launch (the run holds the shared GPU lock and 3 CPU slots: the full-distribution passes use 3 forked
workers; mem_guard at its default limit):

    scripts/gpu_run.sh shared -- scripts/cpu_slot.sh 3 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_cache_build --config experiments/cvtm/configs/cvtm_cache_build.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_lm.py --config experiments/cvtm/configs/cvtm_cache_build.json

Smoke (not logged, scratch numbers only): add `--smoke` (train prefix, a few validation documents,
numpy backend, output under $TMLM_SCRATCH).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
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
from tmlm.cvtm.mixture import COMPONENTS, D_ORDERS, kn_gold_and_depth  # noqa: E402
from tmlm.cvtm.model import CVTMCore, save_kn, load_kn, suffix_gold  # noqa: E402
from tmlm.cvtm.suffix import SuffixClauses  # noqa: E402
from tmlm.eval.baselines.ngram import bits_per_byte  # noqa: E402

LN2 = math.log(2.0)
GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[cvtm_lm {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


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
    """A pipeline check; a failure is logged, recorded and makes the run exit 1 at the end. The details
    are stored under "details" (so no detail can collide with the metric's own fields). In smoke mode a
    reference_check (a comparison with a logged full-data number) is recorded but cannot fail."""
    M(f"check.{name}", bool(ok), details=details)
    if not ok and not (SMOKE and reference_check):
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {details}")


def run_dir(cfg: dict) -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_lm_unlogged"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ memory
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
    """Start only when memory.current + expected peak < 0.80 memory.max (poll every 2 minutes)."""
    waited = 0
    while True:
        cur, ws, mx = cgroup()
        if cur + peak_gib * GIB < 0.80 * mx:
            log(f"memory before {tag}: current {cur / GIB:.1f} GiB + peak {peak_gib:.0f} GiB < 0.80 x {mx / GIB:.0f} GiB "
                f"(waited {waited} s)")
            M(f"memory.wait_s.{tag}", waited)
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag}")
        log(f"memory before {tag}: current {cur / GIB:.1f} GiB + {peak_gib:.0f} GiB >= 0.80 x max; waiting {poll_s} s")
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


# ------------------------------------------------------------------ helpers
def gold_and_check(tag: str, sc: SuffixClauses, kn, streams: dict) -> dict:
    """ln P_S(gold), k* of every stream under one KN variant; checked against KNModel.prob and
    kn_gold_and_depth. Returns {stream: (p, kstar)} with p = exp(ln P_S)."""
    out = {}
    for sname, (tok, ds) in streams.items():
        t0 = time.time()
        lnp, ks = suffix_gold(sc, tok, ds)
        t_vote = time.time() - t0
        t0 = time.time()
        p_ref, k_ref = kn_gold_and_depth(kn, tok, ds)
        dmax = float(np.max(np.abs(lnp - np.log(p_ref))))
        M(f"kn.{tag}.{sname}.max_abs_ln_diff_vote_vs_KNModel", dmax, note="float64 vote sum vs KNModel.prob, every position")
        check(f"kn.{tag}.{sname}.vote_form_within_1e-9", dmax <= 1e-9, max_abs_ln_diff=dmax)
        check(f"kn.{tag}.{sname}.kstar_equals_kn_gold_and_depth", bool(np.array_equal(ks.astype(np.int64), k_ref)))
        M(f"kn.{tag}.{sname}.bits_per_token", float(-lnp.mean() / LN2))
        M(f"kn.{tag}.{sname}.gold_wall_s", {"vote_form": round(t_vote, 1), "KNModel_check": round(time.time() - t0, 1)}, unit="s")
        out[sname] = (np.exp(lnp), ks.astype(np.int8), lnp)
    return out


def boot_rows(name: str, doc_bits: np.ndarray, doc_bytes: np.ndarray, idx: np.ndarray) -> None:
    b = protocol.bootstrap_bpb(doc_bits, doc_bytes, idx)
    M(f"validation.{name}.bits_per_byte_ci95", b["ci95"], label="derived",
      note=f"document-block bootstrap: {b['n_docs']} validation documents, {b['n_boot']} resamples, seed {list(protocol.BOOT_SEED)}; "
           "per resample sum of bits / sum of UTF-8 bytes of the drawn documents")


def sha_of_cvtm_sources() -> dict:
    d = ROOT / "src" / "tmlm" / "cvtm"
    return {p.name: hashing.sha256_file(p) for p in sorted(d.glob("*.py"))}


# ------------------------------------------------------------------ main
def cache_build(cfg: dict, args) -> int:
    rd = run_dir(cfg)
    cdir = rd / "cache"
    if cdir.exists() and any(cdir.iterdir()):
        raise SystemExit(f"{cdir} is not empty")
    cdir.mkdir(parents=True, exist_ok=True)
    p = cfg["params"]
    V = int(p["vocab"])
    k_top = int(p["topk"])
    workers = int(cfg["compute"]["dense_workers"])
    chunk = int(cfg["compute"]["dense_chunk"])
    pool_limit = int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB)
    backend = "numpy" if SMOKE else cfg["compute"]["kn_backend"]
    sampler = PeakSampler()
    M("status", cfg["status"], label="unverified")
    M("config.params", p)

    # ---- disk
    du = shutil.disk_usage(rd)
    need = float(cfg["compute"]["expected_disk_gib"]) * GIB
    M("disk.free_bytes_before", int(du.free), note=f"filesystem of {rd}")
    check("disk.fits_comfortably", need < 0.5 * du.free, expected_bytes=int(need), free_bytes=int(du.free),
          rule="expected cache size below half of the free space")
    if FAILED:
        return 1

    # ---- data
    t0 = time.time()
    M("dataset_hashes", protocol.write_dataset_hashes(("train", "validation"), protocol.CORPUS, V),
      note="sha256 entries of the stores' meta.json files (also in dataset_hashes.json)")
    tok, ds = protocol.load_train(protocol.CORPUS, V, check=not SMOKE)
    ev = protocol.load_eval_split("validation", protocol.CORPUS, V, check=True)
    if SMOKE:
        tok, ds = protocol.prefix_documents(tok, ds, args.smoke_train_tokens)
        kd = args.smoke_val_docs
        vds = ev.doc_start[:kd + 1]
        ev = protocol.EvalSplit(split="validation", tok=ev.tok[:int(vds[-1])], doc_start=vds, doc_bytes=ev.doc_bytes[:kd],
                                n_bytes=int(ev.doc_bytes[:kd].sum()), n_words=0, meta=ev.meta, raw_meta=ev.raw_meta)
    tok = np.ascontiguousarray(np.asarray(tok))
    n_docs = int(ds.shape[0] - 1)
    M("data.train_tokens", int(ds[-1]))
    M("data.train_docs", n_docs)
    M("data.validation_tokens", int(ev.tok.shape[0]))
    M("data.validation_docs", ev.n_docs)
    M("data.validation_bytes_utf8", ev.n_bytes, note="protocol CP byte accounting")
    M("data.load_wall_s", round(time.time() - t0, 1), unit="s")
    M("protocol.selection", protocol.selection_summary(n_docs, ds, protocol.HPRIME_RESIDUES),
      note="residue(i) = mix64(i XOR 0xC0FFEE) mod 100 (docs/12 F20 b); roles 0 core_em, 1 t2_train, 2 selection")
    hp, rest_hp = protocol.split_by_residue(tok, ds, protocol.HPRIME_RESIDUES)
    del rest_hp
    gc.collect()
    streams = {"hprime": (hp.tok, hp.doc_start), "validation": (ev.tok, ev.doc_start)}

    # ---- document clauses (whole-document window)
    t0 = time.time()
    feats = {s: doc_features(t, d, orders=D_ORDERS, window=None, ptr_min=5, ptr_max=64) for s, (t, d) in streams.items()}
    M("doc_clauses.batch_wall_s", round(time.time() - t0, 1), unit="s")
    for s, (t, d) in streams.items():
        same = feats[s].same_values(doc_features_stream(t, d, D_ORDERS, None, 5, 64))
        check(f"doc_clauses.{s}.stream_equals_batch", bool(same))
        M(f"doc_clauses.{s}.n_fallback", int(feats[s].n_fallback))
    files: dict = {}
    gdirs = {s: cdir / s for s in streams}
    for s, (t, d) in streams.items():
        f = feats[s]
        recs = {
            "gold": C.write_array(gdirs[s], "gold", np.asarray(t).astype(np.uint16)),
            "doc_start": C.write_array(gdirs[s], "doc_start", np.asarray(d).astype(np.int64)),
            "doc_index": C.write_array(gdirs[s], "doc_index", hp.doc_index.astype(np.int64) if s == "hprime"
                                       else np.arange(ev.n_docs, dtype=np.int64)),
            "residue": C.write_array(gdirs[s], "residue", hp.residue.astype(np.int8) if s == "hprime"
                                     else np.full(ev.n_docs, -1, dtype=np.int8)),
            "doc_n": C.write_array(gdirs[s], "doc_n", f.n.astype(np.int32)),
            "doc_g": C.write_array(gdirs[s], "doc_g", f.g.astype(np.int32)),
            "ptr_len": C.write_array(gdirs[s], "ptr_len", f.ptr_len.astype(np.int32)),
            "ptr_cand": C.write_array(gdirs[s], "ptr_cand", f.ptr_cand.astype(np.int32)),
            "ptr_pos": C.write_array(gdirs[s], "ptr_pos", f.ptr_pos.astype(np.int64)),
        }
        files[s] = recs
    kn_files: dict = {}
    per: dict = {s: {} for s in streams}          # stream -> variant -> (p, kstar, lnp)
    state: dict = {}

    def save_variant(tag: str, kn, note: str) -> None:
        t0 = time.time()
        rec = save_kn(kn, cdir / C.KN_VARIANTS[tag], extra_meta={"variant": tag, "note": note,
                                                                   "n_train_tokens": int(kn.n_train_tokens)})
        kn_files[tag] = {"file": C.KN_VARIANTS[tag], "sha256": rec["sha256"], "bytes": rec["bytes"],
                         "n_types": [int(x) for x in kn.n_types], "n_train_tokens": int(kn.n_train_tokens),
                         "model_nbytes": kn_gpu.model_nbytes(kn), "note": note}
        state.setdefault("kn_model_hashes", {})[tag] = kn_gpu.model_hashes(kn)
        M(f"kn.{tag}.file", kn_files[tag], note=f"saved in {round(time.time() - t0, 1)} s")

    def write_variant(tag: str, res: dict) -> None:
        for s in streams:
            pk, ks, lnp = res[s]
            per[s][tag] = res[s]
            files[s][f"p_kn_{tag}"] = C.write_array(gdirs[s], f"p_kn_{tag}", pk)
            files[s][f"kstar_{tag}"] = C.write_array(gdirs[s], f"kstar_{tag}", ks)

    def callback(stage: str, **kw) -> None:
        if stage == "kn_noH":
            kn, sc = kw["kn"], kw["suffix"]
            M("kn.noH.fit_info", kn.fit_info)
            M("kn.noH.n_types", [int(x) for x in kn.n_types])
            save_variant("noH", kn, "KN-5 fitted on the training split without H (residue 0)")
            write_variant("noH", gold_and_check("noH", sc, kn, streams))
            return
        if stage == "em":
            weights, fi, trace = kw["weights"], kw["fit_inputs"], kw["trace"]
            state["weights"] = weights
            em_report("full", trace)
            # ---- KN-5 without H' and the H' core arrays (frozen weights over that base)
            _, rest = protocol.split_by_residue(tok, ds, protocol.HPRIME_RESIDUES)
            wait_for_memory(float(cfg["compute"]["kn_peak_gib"]), "kn_noHp_fit")
            t0 = time.time()
            kn = kn_gpu.fit_kn(rest.tok, rest.doc_start, order=5, vocab=V, backend=backend, max_device_bytes=pool_limit)
            del rest
            M("kn.noHp.fit_wall_s", round(time.time() - t0, 1), unit="s")
            M("kn.noHp.fit_info", kn.fit_info)
            M("kn.noHp.n_types", [int(x) for x in kn.n_types])
            sc = SuffixClauses.from_kn(kn, backend="numpy" if backend == "numpy" else "auto")
            save_variant("noHp", kn, "KN-5 fitted on the training split without H' (residues 0, 1, 2)")
            write_variant("noHp", gold_and_check("noHp", sc, kn, streams))
            cfg_core = {"vocab": V, "prefix_tokens": None if not SMOKE else args.smoke_train_tokens, "window": None,
                        "min_counts": None, "components": list(COMPONENTS), "em_iters": 50, "prior": 16.0}
            core_hp = CVTMCore(suffix=sc, weights=weights, config=cfg_core, fit_inputs=fi, kn=None)
            p_hp, k_hp, ln_hp = per["hprime"]["noHp"]
            rec = core_hp.gold(hp.tok, hp.doc_start, lnp_s=ln_hp, kstar=k_hp, feats=feats["hprime"])
            files["hprime"]["bucket"] = C.write_array(gdirs["hprime"], "bucket", rec.bucket.astype(np.int16))
            files["hprime"]["core_p_gold"] = C.write_array(gdirs["hprime"], "core_p_gold", rec.p_gold)
            role = hp.role_of()
            for r, rname in protocol.ROLES.items():
                m = role == r
                M(f"hprime.{rname}.bits_per_token", {"kn5_noHp": float(-np.log2(p_hp[m]).mean()),
                                                     "core_over_noHp": float(-np.log2(rec.p_gold[m]).mean())},
                  note="residue-role positions of H'; base KN-5 without H'; frozen EM weights (residue 0 is the EM's "
                       "own fit documents, with the KN-without-H base there)")
            log("dense pass over H' positions (KN without H' and the core over it)")
            res = C.parallel_dense(core_hp, rec, gdirs["hprime"], k_top, workers, chunk, log)
            files["hprime"].update(res["files"])
            dense_checks("hprime", res["checks"])
            del core_hp, rec, sc, kn, res
            gc.collect()
            return
        if stage == "kn_all":
            kn, sc = kw["kn"], kw["suffix"]
            M("kn.all.fit_info", kn.fit_info)
            M("kn.all.n_types", [int(x) for x in kn.n_types])
            save_variant("all", kn, "KN-5 fitted on ALL training documents (the suffix clauses of the frozen core)")
            write_variant("all", gold_and_check("all", sc, kn, streams))
            return
        raise ValueError(stage)

    def em_report(tag: str, trace: dict) -> None:
        dll = np.diff(np.asarray(trace["pooled_ll"]))
        dob = np.diff(np.asarray(trace["bucket_map_objective"]))
        check(f"em.{tag}.pooled_ll_non_decreasing", bool(dll.min() >= -1e-12), min_step=float(dll.min()))
        check(f"em.{tag}.map_objective_non_decreasing", bool(dob.min() >= -1e-12), min_step=float(dob.min()))
        M(f"em.{tag}.pooled_weights", trace["pooled_weights"])
        M(f"em.{tag}.map_objective_first_last", [trace["bucket_map_objective"][0], trace["bucket_map_objective"][-1]],
          note="mean nats per fit row, iterations 0 and 50")
        M(f"em.{tag}.buckets_with_fit_rows", int(np.count_nonzero(trace["rows_per_bucket"])),
          note=f"of {len(trace['rows_per_bucket'])}")
        M(f"em.{tag}.n_fit_rows", int(trace["n_fit_rows"]))

    def dense_checks(group: str, ch: dict) -> None:
        M(f"dense.{group}.checks", ch)
        check(f"dense.{group}.rows_sum_to_one", ch["base_row_sum_max_abs_err"] < 1e-9 and ch["core_row_sum_max_abs_err"] < 1e-9,
              base=ch["base_row_sum_max_abs_err"], core=ch["core_row_sum_max_abs_err"])
        check(f"dense.{group}.gold_entries_equal_gold_pass", ch["base_gold_max_abs_diff"] == 0.0 and ch["core_gold_max_abs_diff"] == 0.0,
              base=ch["base_gold_max_abs_diff"], core=ch["core_gold_max_abs_diff"])

    # ---- the frozen core (docs/11 11.6)
    wait_for_memory(float(cfg["compute"]["kn_peak_gib"]), "core_fit")
    t0 = time.time()
    core = CVTMCore.fit(tok, ds, vocab=V, prefix_tokens=None, window=None, min_counts=None, components="full",
                        backend=backend, max_device_bytes=pool_limit, callback=callback, log=log,
                        expect_h=None if not SMOKE else False)
    M("core.fit_wall_s_including_callbacks", round(time.time() - t0, 1), unit="s",
      note="includes the cache callbacks (KN without H' fit, H' passes); the core's own phases are in core.fit_info")
    M("core.fit_info", {k: v for k, v in core.fit_info.items() if k != "em_trace"})
    check("core.weights_are_the_em_callback_weights", bool(np.array_equal(core.weights, state["weights"])))
    if not SMOKE:
        check("protocol.H_matches_F20b", bool(core.fit_info["H_checked_against_F20b"]))
    state_hash = core.state_hash()
    M("core.state_hash", state_hash)

    # ---- validation rows
    p_all, k_all, ln_all = per["validation"]["all"]
    rows = {}
    idx = protocol.bootstrap_index(ev.n_docs)
    for name, comps in (("kn5", "S"), ("kn5_D0", "S+D0"), ("cvtm_core", "full")):
        c = core if comps == "full" else core.variant(comps)
        if comps != "full":
            em_report(name, c.fit_info["em_trace"])
        rec = c.gold(ev.tok, ev.doc_start, lnp_s=ln_all, kstar=k_all, feats=feats["validation"])
        sc_ = c.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes, doc_bytes=ev.doc_bytes, rec=rec)
        rows[name] = (sc_, rec)
        M(f"validation.{name}.bits_per_byte", sc_["bits_per_byte"], note="total negative log2-likelihood / UTF-8 bytes (protocol CP)")
        M(f"validation.{name}.bits_per_token", sc_["bits_per_token"])
        M(f"validation.{name}.token_perplexity", 2.0 ** sc_["bits_per_token"], label="derived")
        boot_rows(name, sc_["doc_bits"], ev.doc_bytes, idx)
        if comps == "full":
            files["validation"]["bucket"] = C.write_array(gdirs["validation"], "bucket", rec.bucket.astype(np.int16))
            files["validation"]["core_p_gold"] = C.write_array(gdirs["validation"], "core_p_gold", rec.p_gold)
    db = {n: rows[n][0]["doc_bits"] for n in rows}
    for a, b in (("kn5_D0", "kn5"), ("cvtm_core", "kn5"), ("cvtm_core", "kn5_D0")):
        r = protocol.bootstrap_bpb(db[a], ev.doc_bytes, idx, db[b])
        M(f"validation.{a}.minus_{b}.bits_per_byte", r["point"], label="derived", ci95=r["ci95"],
          note="paired document-block bootstrap (60 documents, 2,000 resamples)")
    tb = float(cfg["reference"]["transformer_600s_validation_bpb"])
    r = protocol.bootstrap_bpb(db["cvtm_core"], ev.doc_bytes, idx, const=tb)
    M("validation.cvtm_core.minus_transformer600s.bits_per_byte", r["point"], label="derived", ci95=r["ci95"],
      note=f"{tb} of {cfg['reference']['transformer_600s_run']} as a constant: the interval covers only the CVTM-core's "
           "document sampling (docs/13 A3 H8.6 placement, no threshold)")

    # ---- pipeline checks against the logged references
    ref = cfg["reference"]
    # (these lines also run in smoke mode, where the comparisons with full-data references cannot fail)
    bpb_knmodel = bits_per_byte(core.kn.logprob(ev.tok, ev.doc_start), ev.n_bytes)
    M("check.kn5_KNModel_path.validation_bits_per_byte", bpb_knmodel)
    check("kn5_KNModel_path_equals_M2_exactly", bpb_knmodel == float(ref["kn5_validation_bpb"]), reference_check=True,
          measured=bpb_knmodel, reference=ref["kn5_validation_bpb"], run=ref["kn5_run"])
    d = abs(rows["kn5"][0]["bits_per_byte"] - float(ref["kn5_validation_bpb"]))
    M("check.kn5_vote_form.abs_diff_vs_M2", d, label="derived")
    check("kn5_vote_form_within_1e-9_of_M2", d <= 1e-9, reference_check=True, abs_diff=d)
    check("kn5_row_7_decimals_1.3501314", round(rows["kn5"][0]["bits_per_byte"], 7) == 1.3501314, reference_check=True,
          measured=rows["kn5"][0]["bits_per_byte"])
    check("kn5_vote_row_equals_KNModel_path_within_1e-12", abs(rows["kn5"][0]["bits_per_byte"] - bpb_knmodel) <= 1e-12,
          abs_diff=abs(rows["kn5"][0]["bits_per_byte"] - bpb_knmodel))
    d = abs(rows["cvtm_core"][0]["bits_per_byte"] - float(ref["cvtm_core_exploratory_bpb"]))
    M("check.cvtm_core.abs_diff_vs_exploratory", d, label="derived", reference=ref["cvtm_core_exploratory_run"])
    check("cvtm_core_within_1e-4_of_exploratory", d <= 1e-4, reference_check=True, abs_diff=d)
    d = abs(rows["cvtm_core"][0]["bits_per_byte"] - float(ref["cvtm_core_mixcheck_bpb"]))
    M("check.cvtm_core.abs_diff_vs_mixture_module_exploratory", d, label="derived", reference=ref["cvtm_core_mixcheck_run"])
    d = abs(rows["kn5_D0"][0]["bits_per_byte"] - float(ref["kn5_D0_mixcheck_bpb"]))
    M("check.kn5_D0.abs_diff_vs_mixture_module_exploratory", d, label="derived", reference=ref["cvtm_core_mixcheck_run"])
    # H8.2 quantities (every validation position; KN on all training documents)
    lq = core.suffix.logprob_stream(ev.tok, ev.doc_start, quantized=True)
    bpb_q = float(-(lq.astype(np.float64) / 65536.0).sum() / LN2 / ev.n_bytes)
    M("h8_2.q16.validation_bits_per_byte", bpb_q, note="integer votes (units of 2^-16 nats) summed exactly, then / ln 2 / bytes")
    M("h8_2.q16_minus_float.validation_bits_per_byte", bpb_q - rows["kn5"][0]["bits_per_byte"], label="derived")
    M("h8_2.max_abs_ln_diff_vote_vs_KNModel", METRICS["kn.all.validation.max_abs_ln_diff_vote_vs_KNModel"]["value"],
      note="same number as kn.all.validation.max_abs_ln_diff_vote_vs_KNModel")
    del lq

    # ---- validation full distributions (frozen core, KN on all training documents)
    rec_core = rows["cvtm_core"][1]
    log("dense pass over validation positions (frozen core)")
    res = C.parallel_dense(core, rec_core, gdirs["validation"], k_top, workers, chunk, log)
    files["validation"].update(res["files"])
    dense_checks("validation", res["checks"])
    top = C.load_arrays(cdir, "validation", ["kn_top_ids", "core_top_ids", "gold"])
    t1_kn = float(np.mean(top["kn_top_ids"][:, 0] == top["gold"]))
    t1_core = float(np.mean(top["core_top_ids"][:, 0] == top["gold"]))
    M("validation.kn5.causal_top1", t1_kn, note="argmax of the full distribution, ties to the lower id")
    M("validation.cvtm_core.causal_top1", t1_core, note="argmax of the full distribution, ties to the lower id")
    if True:                                   # also in smoke mode (plumbing); the values are only meaningful at full data
        M("check.kn5_top1.abs_diff_vs_M2", abs(t1_kn - float(ref["kn5_validation_top1"])), label="derived",
          reference=ref["kn5_run"], note="vote-form distribution vs KNModel.dist argmax; a nonzero value can only come from "
                                         "last-bit ties")
        M("check.cvtm_core_top1.abs_diff_vs_exploratory", abs(t1_core - float(ref["cvtm_core_exploratory_top1"])), label="derived",
          note="the exploratory estimator used another pooled-weight definition (docs/12 F20 c)")
    del top

    # ---- save the frozen core, reload it
    log_ref = rows["cvtm_core"][0]["log2p"].copy()
    t0 = time.time()
    core_file = core.save(cdir / C.CORE_FILE)
    M("core.file", core_file, note=f"saved in {round(time.time() - t0, 1)} s")
    check("core.saved_state_hash_equals_fit", core_file["state_hash"] == state_hash)
    core_config = core.identity()
    core_fit_info = {k: v for k, v in core.fit_info.items() if k != "em_trace"}
    del core, rows, rec_core, res
    gc.collect()
    t0 = time.time()
    core2 = CVTMCore.load(cdir / C.CORE_FILE, backend="numpy" if backend == "numpy" else "auto")
    M("core.reload_wall_s", round(time.time() - t0, 1), unit="s")
    check("core.reload_state_hash_equal", core2.state_hash() == state_hash)
    sc2 = core2.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes)
    check("core.reload_scores_bit_identical", bool(np.array_equal(sc2["log2p"], log_ref)))
    del core2, sc2
    gc.collect()
    for tag in C.KN_VARIANTS:
        t0 = time.time()
        kn_r, meta_r = load_kn(cdir / C.KN_VARIANTS[tag], verify=True)
        check(f"kn.{tag}.reload_model_hashes_equal", kn_gpu.model_hashes(kn_r) == state["kn_model_hashes"][tag])
        M(f"kn.{tag}.reload_wall_s", round(time.time() - t0, 1), unit="s")
        del kn_r
        gc.collect()

    # ---- cache_meta.json
    frecs = {C.KN_VARIANTS[t]: {"sha256": kn_files[t]["sha256"], "bytes": kn_files[t]["bytes"]} for t in C.KN_VARIANTS}
    frecs[C.CORE_FILE] = {"sha256": core_file["sha256"], "bytes": core_file["bytes"]}
    for s in streams:
        for name, r in files[s].items():
            frecs[f"{s}/{name}.npy"] = {"sha256": r["sha256"], "shape": r["shape"], "dtype": r["dtype"]}
    missing = [f"{s}/{n}" for s in streams for n in C.ARRAYS if f"{s}/{n}.npy" not in frecs]
    check("cache.every_array_written", not missing, missing=missing)
    meta = {
        "run_dir": str(rd.relative_to(ROOT)) if rd.is_relative_to(ROOT) else str(rd),
        "smoke": SMOKE,
        "arrays": {n: {"dtype": d, "shape": sh, "content": txt} for n, (d, sh, txt) in C.ARRAYS.items()},
        "groups": {"hprime": {"n": int(hp.n_tokens), "docs": int(hp.n_docs), "core_base": "noHp",
                              "kn_top_base": "noHp", "note": "frozen EM weights over the KN-5 fitted without H'",
                              "held_out": {"residue 0": ["noH", "noHp"], "residues 1 and 2": ["noHp"]},
                              "em_in_sample": "residue 0 (the EM's own fit documents, fitted with the noH base)"},
                   "validation": {"n": int(ev.tok.shape[0]), "docs": int(ev.n_docs), "core_base": "all",
                                  "kn_top_base": "all", "note": "the frozen CVTM-core",
                                  "held_out": {"all positions": ["all", "noH", "noHp"]}}},
        "kn_variants": kn_files,
        "core": {"file": C.CORE_FILE, "sha256": core_file["sha256"], "state_hash": state_hash, "identity": core_config,
                 "fit_info": core_fit_info},
        "protocol": protocol.selection_summary(n_docs, ds, protocol.HPRIME_RESIDUES),
        "dataset_hashes": protocol.dataset_hashes(("train", "validation"), protocol.CORPUS, V),
        "files": frecs,
        "cvtm_sources_sha256": sha_of_cvtm_sources(),
    }
    C.write_meta(cdir, meta)
    M("cache.meta_sha256", hashing.sha256_file(cdir / C.META_FILE))
    M("cache.bytes_total", int(sum(f.stat().st_size for f in cdir.rglob("*") if f.is_file())))
    problems = C.verify(cdir)
    check("cache.verify_rehash", not problems, problems=problems[:10])
    M("freeze.cvtm_sources_sha256", sha_of_cvtm_sources())
    M("peak_rss_gib", round(rss_peak_gib(), 2), note="VmHWM of the driver process (forked dense workers not included)")
    M("peak_cgroup_working_set_gib", round(sampler.stop() / GIB, 2), note="all processes of the container, sampled every 2 s")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    M("checks_failed", list(FAILED))
    return 1 if FAILED else 0


def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train-tokens", type=int, default=3_000_000)
    ap.add_argument("--smoke-val-docs", type=int, default=4)
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    if SMOKE and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py")
    cfg = json.loads(Path(args.config).read_text())
    if cfg["mode"] != "cache_build":
        raise SystemExit(f"unknown mode {cfg['mode']!r}")
    return cache_build(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
