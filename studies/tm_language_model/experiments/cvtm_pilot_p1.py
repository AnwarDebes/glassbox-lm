"""Pilot P1 of docs/13 amendment A5 (EXPLORATORY; supports no claim): CR-TM on the frozen CVTM cache.
Model, literals, gates and read-out: `src/tmlm/cvtm/crtm.py`. Validation and test are NOT used: arm (i)
is selected on H' residue 2a and judged on H' residue 2b; arm (ii) is the BLiMP dev half.

Modes:
- arm1: training pairs (H' residue 1), the 2a and 2b pair stores, then two worker processes (one CPU
  slot each): worker "tm" (TM grid, the two ablations, C0) and worker "ctl" (LR, LR2, HGB). Every gate
  gets the same number of configurations, each scored on 2a with its own pooled (beta, a); the chosen
  configuration gets the per-bucket (beta_b, a_b) on 2a and is judged once on 2b. Then the arm (i)
  contrasts (paired document bootstrap over the 2b documents, 2,000 resamples) and the arm (i) part of
  the A5 verdict. Frozen models are saved for arm (ii).
- blimp-hashes: download BLiMP at the pinned revision into the Hugging Face cache, sha256 every file,
  check it against the repository's LFS oid, and write docs/notes/preregistration-data-hashes-A4-blimp.json
  ONCE (never overwritten; when it exists it is read and verified instead).
- arm2: BLiMP dev half (A4 H9.4 split, `crtm.blimp_dev_mask`), sentence log-probabilities of the frozen
  CVTM-core and of core + each frozen gate of an arm1 run, accuracy, paired bootstrap over pairs, and
  the combined P1 verdict (GO / NO-GO / grey).

Launch lines: see experiments/cvtm/configs/cvtm_pilot_p1_*.json ("launch").
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.cvtm import cache as C  # noqa: E402
from tmlm.cvtm import copy_selector as CS  # noqa: E402
from tmlm.cvtm import crtm  # noqa: E402
from tmlm.cvtm import protocol  # noqa: E402

LN2 = math.log(2.0)
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False
TAG = "p1"
V = protocol.VOCAB
HP_NAMES = ["gold", "doc_start", "doc_index", "residue", "core_top_ids", "core_top_p", "core_p_gold", "bucket",
            "kstar_noHp", "kn_top_ids", "kn_top_p", "ptr_len", "ptr_cand"]


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[p1 {TAG} {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def _plain(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def M(name: str, value, label: str = "measured", **extra) -> None:
    if SMOKE:
        label = "unverified"
    METRICS[name] = {"value": _plain(value), "label": label, "exploratory": True, **_plain(extra)}
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_metrics({name: METRICS[name]}, on_conflict="replace")
    if isinstance(value, (int, float, str, bool, np.floating, np.integer)) or value is None:
        log(f"{name} = {value}")


def check(name: str, ok: bool, **details) -> None:
    M(f"check.{name}", bool(ok), details=details)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {details}")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def out_dir(args) -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(args.work) / f"p1_unlogged_{args.mode}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def limit_threads(n: int = 1) -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception as exc:  # pragma: no cover - reported, not hidden
        log(f"threadpoolctl not applied: {exc!r}")
    try:
        import numba
        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    except Exception as exc:  # pragma: no cover
        log(f"numba threads not set: {exc!r}")


# ------------------------------------------------------------------ shared state of arm (i)
S: dict = {}


def load_hprime(cache_dir: Path, verify_files: bool) -> dict:
    meta = C.read_meta(cache_dir)
    if verify_files:
        bad = []
        for n in HP_NAMES:
            rel = f"hprime/{n}.npy"
            if hashing.sha256_file(cache_dir / rel) != meta["files"][rel]["sha256"]:
                bad.append(rel)
        check("cache.hprime_files_sha256_equal_cache_meta", not bad, differing=bad)
    return C.load_arrays(cache_dir, "hprime", names=HP_NAMES)


def build_sets(arrs: dict, cfg: dict) -> dict:
    ds = np.asarray(arrs["doc_start"]).astype(np.int64)
    residue = np.asarray(arrs["residue"]).astype(np.int64)
    doc_index = np.asarray(arrs["doc_index"]).astype(np.int64)
    n_docs = residue.shape[0]
    a_doc, b_doc = crtm.split_2ab(doc_index, residue)
    r1_doc = residue == 1
    sm = cfg.get("smoke", {}) if SMOKE else {}
    if sm:
        for m in (a_doc, b_doc, r1_doc):
            keep = np.flatnonzero(m)[: int(sm["docs_per_set"])]
            m[:] = False
            m[keep] = True
    out = {"doc_start": ds, "residue": residue, "doc_index": doc_index, "n_docs": n_docs,
           "a_doc": a_doc, "b_doc": b_doc, "r1_doc": r1_doc,
           "pos_r1": crtm.doc_positions(ds, r1_doc), "pos_2a": crtm.doc_positions(ds, a_doc),
           "pos_2b": crtm.doc_positions(ds, b_doc)}
    out["doc_of_pos_all"] = np.repeat(np.arange(n_docs, dtype=np.int64), np.diff(ds))
    return out


def doc_bytes_train() -> np.ndarray:
    from tmlm.data.corpus_store import raw_dir

    off = np.load(raw_dir(protocol.CORPUS, "train") / "doc_bytes.npy").astype(np.int64)
    return np.diff(off)


def train_counts_without_hprime(gold_hprime: np.ndarray) -> np.ndarray:
    tok, _ = protocol.load_train(check=False)
    counts = np.zeros(V, np.int64)
    for lo in range(0, tok.shape[0], 1 << 24):
        counts += np.bincount(np.asarray(tok[lo:lo + (1 << 24)]).astype(np.int64), minlength=V)[:V]
    counts -= np.bincount(np.asarray(gold_hprime).astype(np.int64), minlength=V)[:V]
    if counts.min() < 0:
        raise RuntimeError("H' token counts exceed the train split counts")
    return counts


def gslots_of(arrs: dict, pos: np.ndarray, chunk: int = 131072) -> np.ndarray:
    out = np.empty(pos.shape[0], np.int64)
    for lo in range(0, pos.shape[0], chunk):
        p = pos[lo:lo + chunk]
        out[lo:lo + p.shape[0]] = crtm.gold_slots(np.asarray(arrs["core_top_ids"][p]), np.asarray(arrs["gold"][p]))
    return out


def build_training(arrs: dict, sets: dict, tabs: dict, cfg: dict, seed: int) -> dict:
    t0 = time.time()
    pos_r1 = sets["pos_r1"]
    gs = gslots_of(arrs, pos_r1)
    M("train.residue1_positions", int(pos_r1.shape[0]))
    M("train.residue1_gold_in_top64_share", float((gs >= 0).mean()), label="measured")
    eligible = pos_r1[gs >= 0]
    n_train = int(cfg["train_positions"]) if not SMOKE else int(cfg["smoke"]["train_positions"])
    pos = crtm.sample_positions(eligible, n_train, seed, tag=1)
    M("train.positions", int(pos.shape[0]))
    ds = sets["doc_start"]
    tok = np.asarray(arrs["gold"]).astype(np.int64)
    doc_of = sets["doc_of_pos_all"][pos]
    offset = pos - ds[doc_of]
    doc_id = sets["doc_index"][doc_of]
    K = int(cfg["negatives"])
    feats, labs, cands, slots_all = [], [], [], []
    chunk = 32768
    for lo in range(0, pos.shape[0], chunk):
        pp = pos[lo:lo + chunk]
        pa = crtm.gather(arrs, pp)
        g = crtm.gold_slots(pa.top_ids, pa.gold)
        if np.any(g < 0):
            raise RuntimeError("a sampled training position has no gold among the top-64")
        neg = crtm.draw_negatives(pa.top_p, g, doc_id[lo:lo + chunk], offset[lo:lo + chunk], seed, K)
        slots = np.concatenate([g[:, None], neg], axis=1)
        f, l, c = crtm.pair_features(tok, ds, pa, slots, tabs)
        feats.append(f)
        labs.append(l)
        cands.append(c)
        slots_all.append(slots)
    feat = np.concatenate(feats)
    lab = np.concatenate(labs)
    slots = np.concatenate(slots_all)
    exp_lab = np.tile(np.r_[1, np.zeros(K, np.int64)], pos.shape[0])
    check("train.labels_are_gold_then_K_negatives", bool(np.array_equal(lab.astype(np.int64), exp_lab)))
    check("train.negatives_distinct_and_not_gold",
          bool(all(len(set(r.tolist())) == K + 1 for r in slots[: min(20000, slots.shape[0])])))
    check("train.raw_values_within_RAW_MAX", bool(np.all(feat.max(axis=0) <= crtm.RAW_MAX)),
          max_by_feature={crtm.RAW[i]: int(v) for i, v in enumerate(feat.max(axis=0))})
    M("train.pairs", int(feat.shape[0]))
    M("train.build_s", time.time() - t0)
    return {"pos": pos, "feat": feat, "lab": lab.astype(np.uint32), "slots": slots, "K": K}


def build_store(arrs: dict, sets: dict, tabs: dict, name: str, work: Path) -> dict:
    """Pair features of every top-64 pair of the positions of set `name` (2a or 2b) into a memmap,
    plus the read-out inputs."""
    t0 = time.time()
    pos = sets[f"pos_{name}"]
    q = pos.shape[0]
    ds = sets["doc_start"]
    tok = np.asarray(arrs["gold"]).astype(np.int64)
    fpath = work / f"feat_{name}.npy"
    feat_mm = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.uint8, shape=(q * crtm.TOPK, crtm.NRAW))
    gslot = np.empty(q, np.int64)
    top_p = np.lib.format.open_memmap(work / f"topp_{name}.npy", mode="w+", dtype=np.float64, shape=(q, crtm.TOPK))
    p_gold = np.empty(q)
    bucket = np.empty(q, np.int64)
    slots_one = np.arange(crtm.TOPK, dtype=np.int64)
    fmax = np.zeros(crtm.NRAW, np.int64)
    lab_ok = True
    chunk = 8192
    for lo in range(0, q, chunk):
        pp = pos[lo:lo + chunk]
        pa = crtm.gather(arrs, pp)
        g = crtm.gold_slots(pa.top_ids, pa.gold)
        slots = np.broadcast_to(slots_one, (pp.shape[0], crtm.TOPK))
        f, l, c = crtm.pair_features(tok, ds, pa, slots, tabs)
        feat_mm[lo * crtm.TOPK:(lo + pp.shape[0]) * crtm.TOPK] = f
        fmax = np.maximum(fmax, f.max(axis=0))
        lab2 = l.reshape(-1, crtm.TOPK)
        want = np.zeros_like(lab2)
        hit = g >= 0
        want[np.flatnonzero(hit), g[hit]] = 1
        lab_ok &= bool(np.array_equal(lab2, want))
        gslot[lo:lo + pp.shape[0]] = g
        top_p[lo:lo + pp.shape[0]] = pa.top_p
        p_gold[lo:lo + pp.shape[0]] = pa.p_gold
        bucket[lo:lo + pp.shape[0]] = pa.bucket
        if (lo // chunk) % 25 == 0:
            log(f"  store {name}: {lo + pp.shape[0]}/{q} positions ({time.time() - t0:.0f} s)")
    feat_mm.flush()
    top_p.flush()
    del feat_mm, top_p
    np.save(work / f"gslot_{name}.npy", gslot)
    np.save(work / f"pgold_{name}.npy", p_gold)
    np.save(work / f"bucket_{name}.npy", bucket)
    check(f"store.{name}.labels_equal_gold_slots", lab_ok)
    check(f"store.{name}.raw_values_within_RAW_MAX", bool(np.all(fmax <= crtm.RAW_MAX)))
    M(f"store.{name}.positions", int(q))
    M(f"store.{name}.pairs", int(q * crtm.TOPK))
    M(f"store.{name}.gold_in_top64_share", float((gslot >= 0).mean()))
    M(f"store.{name}.build_s", time.time() - t0)
    return {"q": q}


def load_store(work: Path, name: str):
    feat = np.load(work / f"feat_{name}.npy", mmap_mode="r")
    top_p = np.asarray(np.load(work / f"topp_{name}.npy", mmap_mode="r"))
    gslot = np.load(work / f"gslot_{name}.npy")
    p_gold = np.load(work / f"pgold_{name}.npy")
    bucket = np.load(work / f"bucket_{name}.npy")
    q = gslot.shape[0]
    rd = CS.ReadoutData(offs=np.arange(q + 1, dtype=np.int64) * crtm.TOPK, gslot=gslot, lpgold=np.log(p_gold),
                        bucket=bucket, pc=top_p.reshape(-1))
    return feat, rd


def score_store(feat, fn, chunk_pos: int = 4096) -> np.ndarray:
    n = feat.shape[0]
    out = np.empty(n)
    step = chunk_pos * crtm.TOPK
    for lo in range(0, n, step):
        out[lo:lo + step] = fn(np.asarray(feat[lo:lo + step]))
    return out


# ------------------------------------------------------------------ gate bookkeeping
def pooled_eval(score: np.ndarray, rd) -> dict:
    mu, sd = crtm.standardise(score)
    th, info = CS.fit_pooled((score - mu) / sd, rd)
    return {"mu": mu, "sigma": sd, "theta": th.tolist(), **info}


def finish_gate(gate: str, chosen: dict, score_a: np.ndarray, rd_a, feat_b, rd_b, score_b_fn, run: Path,
                model_blob: dict) -> dict:
    """Per-bucket read-out on 2a for the chosen configuration, 2b scores, saved per-position ln P."""
    t0 = time.time()
    fit = crtm.fit_readout(score_a, rd_a)
    lp_a = fit["lp"]
    score_b = score_store(feat_b, score_b_fn)
    lp_b = crtm.apply_readout(score_b, rd_b, fit)
    np.savez(run / "models" / f"lp_{gate}.npz", lp_2a=lp_a, lp_2b=lp_b)
    blob = dict(model_blob)
    blob.update(readout_mu=fit["mu"], readout_sigma=fit["sigma"], readout_beta=fit["beta"],
                readout_alpha=fit["alpha"], readout_pooled=np.asarray(fit["pooled"]))
    with open(run / "models" / f"gate_{gate}.pkl", "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    rec = {"chosen": chosen, "bits_per_token_2a_per_bucket": float(-lp_a.mean() / LN2),
           "bits_per_token_2b": float(-lp_b.mean() / LN2), "pooled": fit["pooled"],
           "pooled_info": fit["pooled_info"], "bucket_info": {k: v for k, v in fit["bucket_info"].items()
                                                              if k != "pooled_hessian_per_position"},
           "finish_s": time.time() - t0}
    M(f"gate.{gate}.final", rec)
    return rec


# ------------------------------------------------------------------ worker: Tsetlin machines
def rule_text(ls: crtm.LiteralSet, lits: np.ndarray, negs: np.ndarray) -> str:
    return " AND ".join(("NOT " if n else "") + ls.names[int(j)] for j, n in zip(lits, negs)) or "(empty)"


def worker_tm(_arg) -> dict:
    limit_threads(1)
    global TAG
    TAG = "tm"
    from tmu.models.classification.vanilla_classifier import TMClassifier

    cfg, run, work, seed = S["cfg"], S["run"], S["work"], S["seed"]
    tr = S["train"]
    lsets = S["lsets"]
    feat_a, rd_a = load_store(work, "2a")
    feat_b, rd_b = load_store(work, "2b")
    tmc = cfg["tm"] if not SMOKE else {**cfg["tm"], **cfg["smoke"]["tm"]}
    grid = [(int(a), float(b)) for a, b in tmc["grid"]]
    epochs = int(tmc["epochs"])
    ncl = int(tmc["clauses_per_class"])
    summary = {}

    def run_grid(gate: str, ls: crtm.LiteralSet, feat, y, grid_, epochs_, tm_seed: int, verify: bool) -> dict:
        best = None
        recs = []
        for (T, s) in grid_:
            tm = TMClassifier(ncl, T=T, s=s, platform="CPU", weighted_clauses=True, seed=tm_seed)
            for ep in range(1, epochs_ + 1):
                fit_s = crtm.train_tm_epoch(tm, feat, y, ls, tm_seed, ep, chunk=int(tmc["fit_chunk"]))
                cs = CS.tm_clause_set(tm, ls.n)
                ev = crtm.TMEval(ls, cs)
                t1 = time.time()
                sa = score_store(feat_a, ev.score)
                sc_s = time.time() - t1
                if verify:
                    rv = np.random.default_rng(7)
                    ii = np.sort(rv.choice(feat_a.shape[0], size=min(3000, feat_a.shape[0]), replace=False))
                    fa = np.asarray(feat_a[ii])
                    _, csum = tm.predict(np.ascontiguousarray(ls.matrix(fa, np.uint32)), return_class_sums=True)
                    d = csum[:, 1].astype(np.int64) - csum[:, 0].astype(np.int64) - ev.score(fa).astype(np.int64)
                    ud = np.unique(d)
                    check(f"{gate}.T{T}_s{s:g}_e{ep}.evaluator_equals_tmu_class_sum_difference",
                          bool(ud.shape[0] == 1 and int(ud[0]) in (0, cs.empty_const)),
                          distinct_offsets=ud.tolist()[:5], empty_clause_constant=cs.empty_const)
                pe = pooled_eval(sa, rd_a)
                n_inc = np.diff(cs.ptr)
                rec = {"T": T, "s": s, "epoch": ep, "fit_s": fit_s, "score_2a_s": sc_s,
                       "nonempty_clauses": cs.n, "mean_literals_per_nonempty_clause": float(n_inc.mean()) if cs.n else 0.0,
                       "bits_per_token_2a_pooled": pe["bits_per_position"], "pooled_beta": pe["beta"], "pooled_a": pe["a"],
                       "pooled_converged": pe["converged"]}
                recs.append(rec)
                log(f"{gate} T {T} s {s:g} epoch {ep}: 2a pooled {pe['bits_per_position']:.5f} bits/token "
                    f"(core {pe['core_bits_per_position']:.5f}), fit {fit_s:.0f} s, score {sc_s:.0f} s, "
                    f"{cs.n} non-empty clauses, rss {rss_gib():.2f} GiB")
                if best is None or pe["bits_per_position"] < best["rec"]["bits_per_token_2a_pooled"]:
                    best = {"rec": rec, "cs": cs, "score_a": sa}
                else:
                    del sa
                gc.collect()
            del tm
            gc.collect()
        M(f"gate.{gate}.configs", recs)
        return best

    # ---- TM on MAIN
    t0 = time.time()
    best = run_grid("TM", lsets["MAIN"], tr["feat"], tr["lab"], grid, epochs, seed, verify=True)
    cs = best["cs"]
    ev = crtm.TMEval(lsets["MAIN"], cs)
    summary["TM"] = finish_gate("TM", best["rec"], best["score_a"], rd_a, feat_b, rd_b, ev.score, run,
                                {"kind": "tm", "literal_set": "MAIN", "ptr": cs.ptr, "lits": cs.lits, "negs": cs.negs,
                                 "w": cs.w, "empty_const": cs.empty_const, "config": best["rec"]})
    # readable clauses: the largest |weight|, firing rate and precision on 2a pairs (first 400,000 pairs)
    sub = np.asarray(feat_a[: min(400000, feat_a.shape[0])])
    labs_sub = np.zeros(sub.shape[0], np.uint8)
    gs = rd_a.gslot[: sub.shape[0] // crtm.TOPK]
    hit = np.flatnonzero(gs >= 0)
    labs_sub[hit * crtm.TOPK + gs[hit]] = 1
    order = np.argsort(-np.abs(cs.w), kind="stable")[:15]
    rules = []
    for ci in order:
        one = CS.ClauseSet(ptr=np.array([0, cs.ptr[ci + 1] - cs.ptr[ci]], np.int64), lits=cs.lits[cs.ptr[ci]:cs.ptr[ci + 1]],
                           negs=cs.negs[cs.ptr[ci]:cs.ptr[ci + 1]], w=np.array([1], np.int64), origin=cs.origin[ci:ci + 1])
        fire = crtm.TMEval(lsets["MAIN"], one).score(sub) > 0
        rules.append({"signed_weight": int(cs.w[ci]), "class_bank": int(cs.origin[ci, 0]),
                      "rule": rule_text(lsets["MAIN"], one.lits, one.negs), "n_literals": int(one.lits.shape[0]),
                      "pair_firing_rate_2a": float(fire.mean()),
                      "precision_gold_2a": float(labs_sub[fire].mean()) if fire.any() else None})
    M("gate.TM.top15_clauses", rules, note="largest |signed weight| (class 1 votes +, class 0 votes -); rates on the first 400,000 2a pairs")
    # literal families used by the chosen clauses
    fam = {}
    for j in cs.lits:
        f = lsets["MAIN"].family[int(j)]
        fam[f] = fam.get(f, 0) + 1
    M("gate.TM.included_literals_by_family", fam)
    M("gate.TM.wall_s", time.time() - t0)
    del best
    gc.collect()

    # ---- ablations at the chosen (T, s, epoch)
    ch = summary["TM"]["chosen"]
    for gate, lsname in (("TM_norel", "NOREL"), ("TM_ctxid", "CTXID")):
        t0 = time.time()
        b = run_grid(gate, lsets[lsname], tr["feat"], tr["lab"], [(ch["T"], ch["s"])], ch["epoch"], seed, verify=False)
        summary[gate] = finish_gate(gate, b["rec"], b["score_a"], rd_a, feat_b, rd_b,
                                    crtm.TMEval(lsets[lsname], b["cs"]).score, run,
                                    {"kind": "tm", "literal_set": lsname, "ptr": b["cs"].ptr, "lits": b["cs"].lits,
                                     "negs": b["cs"].negs, "w": b["cs"].w, "empty_const": b["cs"].empty_const,
                                     "config": b["rec"], "note": "chosen TM configuration, epochs up to the chosen epoch (the best of them on 2a)"})
        M(f"gate.{gate}.wall_s", time.time() - t0)
        del b
        gc.collect()

    # ---- C0: equal size, 1 percent of the pairs (1 percent of the training positions, all their pairs)
    t0 = time.time()
    npos = tr["pos"].shape[0]
    K1 = tr["K"] + 1
    n_c0 = max(1, int(round(float(cfg["c0_fraction"]) * npos)))
    pick = np.sort(np.random.default_rng(np.random.SeedSequence([seed, 0xC0, 1])).choice(npos, size=n_c0, replace=False))
    rows = (pick[:, None] * K1 + np.arange(K1)[None, :]).reshape(-1)
    M("gate.C0.pairs", int(rows.shape[0]))
    b = run_grid("C0", lsets["MAIN"], tr["feat"][rows], tr["lab"][rows], grid, epochs, seed, verify=False)
    summary["C0"] = finish_gate("C0", b["rec"], b["score_a"], rd_a, feat_b, rd_b, crtm.TMEval(lsets["MAIN"], b["cs"]).score,
                                run, {"kind": "tm", "literal_set": "MAIN", "ptr": b["cs"].ptr, "lits": b["cs"].lits,
                                      "negs": b["cs"].negs, "w": b["cs"].w, "empty_const": b["cs"].empty_const,
                                      "config": b["rec"]})
    M("gate.C0.wall_s", time.time() - t0)
    M("worker.tm.peak_rss_gib", rss_gib())
    return summary


# ------------------------------------------------------------------ worker: controls
def worker_ctl(_arg) -> dict:
    limit_threads(1)
    global TAG
    TAG = "ctl"
    cfg, run, work, seed = S["cfg"], S["run"], S["work"], S["seed"]
    tr = S["train"]
    ls = S["lsets"]["MAIN"]
    feat_a, rd_a = load_store(work, "2a")
    feat_b, rd_b = load_store(work, "2b")
    cc = cfg["controls"] if not SMOKE else {**cfg["controls"], **cfg["smoke"]["controls"]}
    summary = {}
    # ---- LR and LR2
    for gate, pairwise, maxiter in (("LR", False, int(cc["lr_maxiter"])), ("LR2", True, int(cc["lr2_maxiter"]))):
        t0 = time.time()
        model = crtm.TableLR(ls, pairwise)
        M(f"gate.{gate}.n_params", model.n_params)
        best, recs, theta = None, [], None
        for l2 in [float(x) for x in cc["l2_grid"]]:           # descending: warm start from the previous
            theta, info = model.fit(tr["feat"], tr["lab"], l2, theta0=theta, maxiter=maxiter)
            T = model.tables(theta)
            sa = score_store(feat_a, lambda f: model.score(f, theta, T))
            pe = pooled_eval(sa, rd_a)
            rec = {"l2": l2, **{k: info[k] for k in ("nit", "converged", "stopped_at_cap", "fit_s", "loss", "maxiter")},
                   "bits_per_token_2a_pooled": pe["bits_per_position"], "pooled_beta": pe["beta"], "pooled_a": pe["a"]}
            recs.append(rec)
            log(f"{gate} l2 {l2:g}: 2a pooled {pe['bits_per_position']:.5f} bits/token, nit {info['nit']} "
                f"(cap {maxiter}{', AT CAP' if info['stopped_at_cap'] else ''}), fit {info['fit_s']:.0f} s, rss {rss_gib():.2f} GiB")
            if best is None or pe["bits_per_position"] < best["rec"]["bits_per_token_2a_pooled"]:
                best = {"rec": rec, "theta": theta.copy(), "score_a": sa}
            else:
                del sa
            gc.collect()
        if best["rec"]["stopped_at_cap"] and bool(cc.get("refit_capped_chosen", True)):
            cap4 = 4 * maxiter
            theta, info = model.fit(tr["feat"], tr["lab"], best["rec"]["l2"], theta0=best["theta"], maxiter=cap4)
            T = model.tables(theta)
            sa = score_store(feat_a, lambda f: model.score(f, theta, T))
            pe = pooled_eval(sa, rd_a)
            rec = {"l2": best["rec"]["l2"], "refit_of_capped_chosen": True,
                   **{k: info[k] for k in ("nit", "converged", "stopped_at_cap", "fit_s", "loss", "maxiter")},
                   "bits_per_token_2a_pooled": pe["bits_per_position"]}
            recs.append(rec)
            log(f"{gate} refit at 4x cap: 2a pooled {pe['bits_per_position']:.5f}, nit {info['nit']}, at cap {info['stopped_at_cap']}")
            best = {"rec": rec, "theta": theta.copy(), "score_a": sa}
        M(f"gate.{gate}.configs", recs)
        th = best["theta"]
        Tb = model.tables(th)
        summary[gate] = finish_gate(gate, best["rec"], best["score_a"], rd_a, feat_b, rd_b,
                                    lambda f: model.score(f, th, Tb), run,
                                    {"kind": "table_lr", "pairwise": pairwise, "literal_set": "MAIN", "theta": th,
                                     "config": best["rec"]})
        M(f"gate.{gate}.wall_s", time.time() - t0)
        del best, model
        gc.collect()
    # ---- HGB
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    lay = crtm.hgb_layout(ls)
    M("gate.HGB.columns", len(lay.cols))
    M("gate.HGB.categorical_columns", int(lay.categorical.sum()))
    X = crtm.hgb_matrix(tr["feat"], lay, dtype=np.float32)     # small integers: exact in float32; sklearn makes its float64 copy
    y = tr["lab"].astype(np.int64)
    best, recs = None, []
    stages = [int(s) for s in cc["hgb_stages"]]
    for lr in [float(x) for x in cc["hgb_learning_rates"]]:
        t1 = time.time()
        mdl = HistGradientBoostingClassifier(learning_rate=lr, max_iter=max(stages), max_leaf_nodes=int(cc["hgb_max_leaf_nodes"]),
                                             min_samples_leaf=int(cc["hgb_min_samples_leaf"]),
                                             l2_regularization=float(cc["hgb_l2"]), categorical_features=lay.categorical,
                                             early_stopping=False, random_state=seed)
        mdl.fit(X, y)
        fit_s = time.time() - t1
        trees = crtm.hgb_trees(mdl, lay)
        rv = np.random.default_rng(7)
        ii = np.sort(rv.choice(feat_a.shape[0], size=min(3000, feat_a.shape[0]), replace=False))
        fa = np.asarray(feat_a[ii])
        mine = crtm.hgb_staged_scores(trees, fa, [max(stages)])[0]
        ref = mdl.decision_function(crtm.hgb_matrix(fa, lay))
        check(f"HGB.lr{lr:g}.walker_equals_decision_function", bool(np.array_equal(mine, ref)),
              max_abs_diff=float(np.max(np.abs(mine - ref))))
        st_scores = crtm.hgb_staged_scores(trees, feat_a, stages)
        for k, stg in enumerate(stages):
            sa = st_scores[k]
            pe = pooled_eval(sa, rd_a)
            rec = {"learning_rate": lr, "trees": stg, "fit_s": fit_s, "bits_per_token_2a_pooled": pe["bits_per_position"],
                   "pooled_beta": pe["beta"], "pooled_a": pe["a"]}
            recs.append(rec)
            log(f"HGB lr {lr:g} trees {stg}: 2a pooled {pe['bits_per_position']:.5f} bits/token, fit {fit_s:.0f} s, rss {rss_gib():.2f} GiB")
            if best is None or pe["bits_per_position"] < best["rec"]["bits_per_token_2a_pooled"]:
                best = {"rec": rec, "model": mdl, "trees": trees, "score_a": sa.copy()}
        del st_scores
        gc.collect()
    del X
    M("gate.HGB.configs", recs)
    mdl = best["model"]
    trees_b = best["trees"]
    stg = best["rec"]["trees"]
    summary["HGB"] = finish_gate("HGB", best["rec"], best["score_a"], rd_a, feat_b, rd_b,
                                 lambda f: crtm.hgb_staged_scores(trees_b, f, [stg])[0], run,
                                 {"kind": "hgb", "literal_set": "MAIN", "model": mdl, "trees": stg, "config": best["rec"]})
    M("gate.HGB.wall_s", time.time() - t0)
    M("worker.ctl.peak_rss_gib", rss_gib())
    return summary


# ------------------------------------------------------------------ arm (i)
def arm1(args, cfg: dict) -> int:
    seed = int(cfg["seed"])
    run = out_dir(args)
    (run / "models").mkdir(parents=True, exist_ok=True)
    work = Path(args.work) / (f"p1_seed{seed}_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ("_smoke" if SMOKE else ""))
    work.mkdir(parents=True, exist_ok=True)
    M("status", "EXPLORATORY pilot P1 of docs/13 A5 (arm i); supports no claim; validation and test are not used", label="unverified")
    M("config", cfg, label="unverified")
    M("work_dir", str(work), label="unverified", note="scratch pair stores, deleted at the end unless --keep-work")
    cache_dir = ROOT / cfg["cache_run"] / "cache"
    arrs = load_hprime(cache_dir, verify_files=not SMOKE)
    if runlog.current_run_dir() is not None and not SMOKE:
        meta = C.read_meta(cache_dir)
        runlog.write_dataset_hashes({f"cache_hprime_{n}": meta["files"][f"hprime/{n}.npy"]["sha256"] for n in HP_NAMES},
                                    on_conflict="replace")
        protocol.write_dataset_hashes(("train",))
        from tmlm.data import bpe
        from tmlm.data.corpus_store import split_dir as _sd

        tmeta = json.loads((_sd(protocol.CORPUS, V, "train") / "meta.json").read_text())
        runlog.write_dataset_hashes({"train_kind_map_V8000": tmeta["sha256"]["kind_map.npy"],
                                     "tokenizer_bpe_8000": bpe.tokenizer_sha256(8000),
                                     "train_doc_bytes": json.loads((ROOT / "data" / protocol.CORPUS / "raw" / "train" / "meta.json").read_text())["sha256"]["doc_bytes.npy"]},
                                    on_conflict="replace")
    sets = build_sets(arrs, cfg)
    for nm in ("r1", "2a", "2b"):
        dm = sets[{"r1": "r1_doc", "2a": "a_doc", "2b": "b_doc"}[nm]]
        M(f"data.{nm}.documents", int(dm.sum()))
        M(f"data.{nm}.positions", int(sets[f"pos_{nm}"].shape[0]))
    check("data.2a_2b_disjoint_and_residue_2", bool(not np.any(sets["a_doc"] & sets["b_doc"])
                                                     and np.all(sets["residue"][sets["a_doc"] | sets["b_doc"]] == 2)))
    if not SMOKE:
        check("data.2a_130_and_2b_131_documents", bool(sets["a_doc"].sum() == 130 and sets["b_doc"].sum() == 131))
    M("data.2a_doc_index_sha256", protocol.index_sha256(sets["doc_index"][sets["a_doc"]]))
    M("data.2b_doc_index_sha256", protocol.index_sha256(sets["doc_index"][sets["b_doc"]]))
    # token tables and literals
    from tmlm.data.corpus_store import split_dir

    km = np.load(split_dir(protocol.CORPUS, V, "train") / "kind_map.npy")
    counts = train_counts_without_hprime(arrs["gold"])
    tabs = crtm.token_tables(km, counts, ROOT / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json")
    all_lits = crtm.build_literals(tabs["c2_used"])
    lsets = crtm.literal_sets(all_lits, int(tabs["idx4"].max()))
    for k, ls in lsets.items():
        fam = {}
        for f in ls.family:
            fam[f] = fam.get(f, 0) + 1
        M(f"literals.{k}.n", ls.n)
        M(f"literals.{k}.by_family", fam)
    M("literals.MAIN.names", lsets["MAIN"].names, label="unverified")
    # training pairs and stores
    tr = build_training(arrs, sets, tabs, cfg, seed)
    for name in ("2a", "2b"):
        build_store(arrs, sets, tabs, name, work)
    # document bookkeeping of 2a and 2b
    dbytes_all = doc_bytes_train()
    docs = {}
    for name, dm in (("2a", sets["a_doc"]), ("2b", sets["b_doc"])):
        dl = np.flatnonzero(dm)
        ln = np.diff(sets["doc_start"])[dl]
        docs[name] = {"doc_of": np.repeat(np.arange(dl.shape[0]), ln), "n": int(dl.shape[0]),
                      "bytes": dbytes_all[sets["doc_index"][dl]].astype(np.float64)}
        M(f"data.{name}.utf8_bytes", float(docs[name]["bytes"].sum()))
    del arrs
    gc.collect()
    M("parent.rss_before_workers_gib", rss_gib())
    # ---- models: two workers (one CPU slot each)
    S.update(cfg=cfg, run=run, work=work, seed=seed, train=tr, lsets=lsets)
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(1 if SMOKE else 2) as pool:          # the smoke runs one worker at a time (one CPU)
        r_tm = pool.apply_async(worker_tm, (None,))
        r_ctl = pool.apply_async(worker_ctl, (None,))
        res_tm = r_tm.get()
        res_ctl = r_ctl.get()
    M("models.wall_s", time.time() - t0)
    gates = {**res_tm, **res_ctl}
    # ---- judgement on 2b
    _, rd_b = load_store(work, "2b")
    _, rd_a = load_store(work, "2a")
    lp = {"core": rd_b.lpgold}
    lpa = {"core": rd_a.lpgold}
    for g in gates:
        z = np.load(run / "models" / f"lp_{g}.npz")
        lp[g] = z["lp_2b"]
        lpa[g] = z["lp_2a"]
    idx = protocol.bootstrap_index(docs["2b"]["n"], protocol.N_BOOT, protocol.BOOT_SEED)
    dbits = {g: crtm.doc_bits(v, docs["2b"]["doc_of"], docs["2b"]["n"]) for g, v in lp.items()}
    by = docs["2b"]["bytes"]
    for g in dbits:
        M(f"2b.{g}.bits_per_byte", float(dbits[g].sum() / by.sum()), label="measured")
        M(f"2a.{g}.bits_per_byte_in_sample", float(crtm.doc_bits(lpa[g], docs["2a"]["doc_of"], docs["2a"]["n"]).sum()
                                                   / docs["2a"]["bytes"].sum()), label="measured",
          note="2a is the selection set (per-bucket read-out fitted on it): in-sample")
    ctrl_2a = {g: float(crtm.doc_bits(lpa[g], docs["2a"]["doc_of"], docs["2a"]["n"]).sum() / docs["2a"]["bytes"].sum())
               for g in ("LR", "LR2", "HGB")}
    best_ctrl = min(ctrl_2a, key=ctrl_2a.get)
    M("arm1.best_control_chosen_on_2a", best_ctrl, label="derived", bits_per_byte_2a=ctrl_2a)
    diffs = {}
    for a, b in [("TM", "core"), ("TM", best_ctrl), ("TM", "C0"), ("TM", "LR"), ("TM", "LR2"), ("TM", "HGB"),
                 ("LR", "core"), ("LR2", "core"), ("HGB", "core"), ("C0", "core"), ("TM_norel", "core"),
                 ("TM_ctxid", "core"), ("TM", "TM_norel"), ("TM", "TM_ctxid")]:
        d = crtm.contrast(dbits[a], dbits[b], by, idx)
        diffs[(a, b)] = d
        M(f"2b.contrast.{a}_minus_{b}", d, label="derived",
          note="bits per byte, a minus b (negative = a better); paired document bootstrap over the 2b documents, 2,000 resamples")
    v1 = crtm.verdict_arm1(diffs[("TM", "core")], diffs[("TM", best_ctrl)], diffs[("TM", "C0")])
    M("verdict.arm_i", v1, label="derived", best_control=best_ctrl)
    np.savez_compressed(run / "per_position_2b.npz", **{f"lp_{g}": v for g, v in lp.items()},
                        doc_of=docs["2b"]["doc_of"].astype(np.int32), doc_bytes=by)
    M("wall_s", time.time() - T0)
    M("failed_checks", FAILED, label="measured")
    if not args.keep_work:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        M("work_dir_removed", True, label="measured")
    return 1 if FAILED else 0


# ------------------------------------------------------------------ BLiMP data (A4 new data, H9.4 split)
BLIMP_HASH_FILE = ROOT / "docs" / "notes" / "preregistration-data-hashes-A4-blimp.json"


def _hf_cache() -> Path:
    return Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"


def blimp_download() -> dict:
    """Every parquet file of the pinned BLiMP revision in the default Hugging Face cache, with its sha256
    and the repository's LFS oid (the sha256 the hub records)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries = [e for e in api.list_repo_tree(crtm.BLIMP_REPO, repo_type="dataset", revision=crtm.BLIMP_REVISION,
                                             recursive=True, expand=True) if str(e.path).endswith(".parquet")]
    out = {}
    for e in sorted(entries, key=lambda x: x.path):
        path = hf_hub_download(crtm.BLIMP_REPO, e.path, repo_type="dataset", revision=crtm.BLIMP_REVISION,
                               cache_dir=str(_hf_cache()))
        sha = hashing.sha256_file(Path(path))
        lfs = getattr(e, "lfs", None)
        lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs is not None else None
        out[str(e.path)] = {"local_path": str(path), "sha256": sha, "bytes": int(Path(path).stat().st_size),
                            "hub_lfs_sha256": lfs_sha, "hub_lfs_sha256_equal": bool(lfs_sha == sha)}
    return out


def blimp_local(files: dict) -> dict:
    """Local paths of the recorded files (no network), re-hashed."""
    from huggingface_hub import hf_hub_download

    out = {}
    for rel, rec in files.items():
        path = hf_hub_download(crtm.BLIMP_REPO, rel, repo_type="dataset", revision=crtm.BLIMP_REVISION,
                               cache_dir=str(_hf_cache()), local_files_only=True)
        out[rel] = {"local_path": str(path), "sha256": hashing.sha256_file(Path(path))}
    return out


def blimp_rows(paths: list[str]) -> dict:
    import pyarrow.parquet as pq

    cols = {"UID": [], "pair_id": [], "sentence_good": [], "sentence_bad": [], "field": [], "linguistics_term": []}
    for p in sorted(paths):
        t = pq.read_table(p).to_pydict()
        for k in cols:
            cols[k].extend(t[k])
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["pair_id"] = out["pair_id"].astype(np.int64)
    return out


def pair_list_sha256(uid: np.ndarray, pid: np.ndarray) -> str:
    import hashlib

    order = np.lexsort((pid, uid))
    txt = "".join(f"{uid[i]}\t{int(pid[i])}\n" for i in order)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def blimp_hashes(args, cfg: dict) -> int:
    """Write docs/notes/preregistration-data-hashes-A4-blimp.json once, or read and verify it."""
    from tmlm.data import bpe

    M("status", "BLiMP data hashes for docs/13 A4 (new data) and the H9.4 dev/report split used by P1 arm (ii)", label="unverified")
    files = blimp_download()
    for rel, rec in files.items():
        check(f"blimp.{rel}.sha256_equals_hub_lfs_oid", rec["hub_lfs_sha256_equal"], sha256=rec["sha256"], hub=rec["hub_lfs_sha256"])
    rows = blimp_rows([r["local_path"] for r in files.values()])
    n = rows["UID"].shape[0]
    dev = crtm.blimp_dev_mask(rows["UID"], rows["pair_id"])
    per_uid = {u: int(dev[rows["UID"] == u].sum()) for u in np.unique(rows["UID"])}
    check("blimp.67_paradigms_of_1000_pairs", bool(len(per_uid) == 67 and n == 67000
                                                     and all(int((rows["UID"] == u).sum()) == 1000 for u in per_uid)))
    check("blimp.dev_half_is_500_per_paradigm", bool(all(v == 500 for v in per_uid.values())))
    check("blimp.pairID_unique_within_paradigm", bool(all(np.unique(rows["pair_id"][rows["UID"] == u]).shape[0] == 1000 for u in per_uid)))
    record = {
        "purpose": "docs/13 amendment A4, new data: BLiMP (67,000 pairs), hashed before use; the H9.4 split (half of each paradigm's pairs are dev, the other half is reported once) as used by pilot P1 arm (ii) of A5",
        "written_by": "experiments/cvtm_pilot_p1.py blimp-hashes (lane P1 of docs/13 A5); write-once, never edited",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"repo": crtm.BLIMP_REPO, "repo_type": "dataset", "revision": crtm.BLIMP_REVISION, "license": "cc-by-4.0",
                   "access": "huggingface_hub.hf_hub_download into the default Hugging Face cache (~/.cache/huggingface/hub)"},
        "files": {rel: {"sha256": r["sha256"], "bytes": r["bytes"], "hub_lfs_sha256_equal": r["hub_lfs_sha256_equal"]}
                  for rel, r in files.items()},
        "n_files": len(files), "n_pairs": int(n),
        "split_H9_4": {"rule": "src/tmlm/cvtm/crtm.py blimp_dev_mask: inside each paradigm (UID) the pairs are ordered by mix64(first 8 bytes (little endian) of sha256(UID) XOR pair_id XOR 0xB11A9D), ties by pair_id; the first floor(n / 2) are dev, the rest are the report half (reported once)",
                       "n_dev_pairs": int(dev.sum()), "n_report_pairs": int((~dev).sum()),
                       "dev_pairs_sha256": pair_list_sha256(rows["UID"][dev], rows["pair_id"][dev]),
                       "report_pairs_sha256": pair_list_sha256(rows["UID"][~dev], rows["pair_id"][~dev]),
                       "pair_list_format": "lines 'UID<TAB>pair_id<LF>' sorted by (UID, pair_id), UTF-8"},
        "preprocessing": "each sentence is one document: crtm.wikitext_style (leading space; clitics n't, 's, 're, 've, 'll, 'd, 'm and a bare trailing ' split off; . , ? ! ; : split off; single spaces), then the BPE 8000 tokenizer of data/tokenizers/bpe_8000",
        "tokenizer_sha256": bpe.tokenizer_sha256(8000),
    }
    if BLIMP_HASH_FILE.exists():
        old = json.loads(BLIMP_HASH_FILE.read_text())
        same_files = old.get("files", {}).keys() == record["files"].keys() and all(
            old["files"][k]["sha256"] == record["files"][k]["sha256"] for k in record["files"])
        check("blimp.hash_file_exists_and_files_match", same_files)
        check("blimp.hash_file_split_matches", old.get("split_H9_4", {}).get("dev_pairs_sha256") == record["split_H9_4"]["dev_pairs_sha256"],
              recorded=old.get("split_H9_4", {}).get("rule"))
        M("blimp.hash_file", "existing file read and verified; not rewritten", label="measured")
    else:
        with open(BLIMP_HASH_FILE, "x", encoding="utf-8") as fh:        # exclusive create: never overwrites
            json.dump(record, fh, indent=1, sort_keys=True)
            fh.write("\n")
        M("blimp.hash_file", f"written once: {BLIMP_HASH_FILE.relative_to(ROOT)}", label="measured",
          sha256=hashing.sha256_file(BLIMP_HASH_FILE))
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({f"blimp_{rel.split('/')[0]}": r["sha256"] for rel, r in files.items()}, on_conflict="replace")
    return 1 if FAILED else 0


# ------------------------------------------------------------------ arm (ii)
def load_gates(arm1_run: Path, lsets: dict) -> dict:
    gates = {}
    for p in sorted((arm1_run / "models").glob("gate_*.pkl")):
        name = p.stem[len("gate_"):]
        with open(p, "rb") as fh:
            blob = pickle.load(fh)
        ls = lsets[blob["literal_set"]]
        if blob["kind"] == "tm":
            cs = CS.ClauseSet(ptr=blob["ptr"], lits=blob["lits"], negs=blob["negs"], w=blob["w"],
                              origin=np.zeros((blob["w"].shape[0], 2), np.int64), empty_const=int(blob["empty_const"]))
            fn = crtm.TMEval(ls, cs).score
        elif blob["kind"] == "table_lr":
            model = crtm.TableLR(ls, bool(blob["pairwise"]))
            th = blob["theta"]
            T = model.tables(th)
            fn = (lambda m, t, T_: (lambda f: m.score(f, t, T_)))(model, th, T)
        elif blob["kind"] == "hgb":
            lay = crtm.hgb_layout(ls)
            trees = crtm.hgb_trees(blob["model"], lay)
            fn = (lambda tr_, st: (lambda f: crtm.hgb_staged_scores(tr_, f, [st])[0]))(trees, int(blob["trees"]))
        else:
            raise ValueError(f"unknown gate kind {blob['kind']}")
        fit = {"mu": float(blob["readout_mu"]), "sigma": float(blob["readout_sigma"]),
               "beta": np.asarray(blob["readout_beta"]), "alpha": np.asarray(blob["readout_alpha"])}
        gates[name] = {"score": fn, "fit": fit, "config": blob.get("config")}
    return gates


def smoke_core(log_fn):
    """SMOKE ONLY: a small CVTM-core fitted on the first 3M training tokens (CPU backends), so the arm (ii)
    pipeline can be exercised under the smoke memory limit. Its numbers are never reported."""
    from tmlm.cvtm.model import CVTMCore

    tok, ds = protocol.load_train(check=False)
    return CVTMCore.fit(tok, ds, prefix_tokens=3_000_000, backend="numpy", expect_h=False, log=log_fn)


def arm2(args, cfg: dict) -> int:
    from tmlm.data import bpe
    from tmlm.data.corpus_store import split_dir

    run = out_dir(args)
    arm1_run = Path(args.arm1_run)
    M("status", "EXPLORATORY pilot P1 of docs/13 A5, arm (ii): BLiMP dev half; supports no claim; validation and test are not used", label="unverified")
    M("config", cfg, label="unverified")
    M("arm1_run", str(arm1_run), label="unverified")
    if not BLIMP_HASH_FILE.exists():
        raise SystemExit(f"{BLIMP_HASH_FILE} does not exist: run the blimp-hashes mode first (A4: hash before use)")
    rec_h = json.loads(BLIMP_HASH_FILE.read_text())
    loc = blimp_local(rec_h["files"])
    check("blimp.files_equal_hash_file", all(loc[k]["sha256"] == rec_h["files"][k]["sha256"] for k in rec_h["files"]))
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_dataset_hashes({f"blimp_{k.split('/')[0]}": v["sha256"] for k, v in rec_h["files"].items()}, on_conflict="replace")
    rows = blimp_rows([v["local_path"] for v in loc.values()])
    dev = crtm.blimp_dev_mask(rows["UID"], rows["pair_id"])
    check("blimp.dev_split_equals_hash_file", pair_list_sha256(rows["UID"][dev], rows["pair_id"][dev]) == rec_h["split_H9_4"]["dev_pairs_sha256"])
    di = np.flatnonzero(dev)
    if SMOKE:
        di = di[: int(cfg["smoke"]["blimp_pairs"])]
    M("blimp.dev_pairs_scored", int(di.shape[0]))
    texts = []
    for i in di:
        texts += [crtm.wikitext_style(str(rows["sentence_good"][i])), crtm.wikitext_style(str(rows["sentence_bad"][i]))]
    tokzr = bpe.load_tokenizer(8000, ROOT / "data")
    tok, ds = bpe.encode_docs(tokzr, texts)
    tok = tok.astype(np.int64)
    ds = ds.astype(np.int64)
    M("blimp.tokens", int(tok.shape[0]))
    M("blimp.example", {"good": texts[0], "bad": texts[1], "good_ids": tok[ds[0]:ds[1]].tolist()}, label="unverified")
    # frozen core (or the smoke core), gold pass and the top-64 of the core and of P_S
    t0 = time.time()
    if SMOKE:
        core = smoke_core(log)
    else:
        core = C.load_core(ROOT / cfg["cache_run"] / "cache", backend="numpy")
        M("core.state_hash", core.state_hash())
    M("core.load_s", time.time() - t0)
    t0 = time.time()
    rec = core.gold(tok, ds)
    dense = core.dense_pass(rec, crtm.TOPK, want_base=True, want_core=True, chunk=2048)
    ch = dense["checks"]
    check("blimp.dense_rows_sum_to_one", bool(ch["core_row_sum_max_abs_err"] < 1e-9 and ch["base_row_sum_max_abs_err"] < 1e-9), **ch)
    check("blimp.dense_gold_equals_gold_pass", bool(ch["core_gold_max_abs_diff"] == 0.0), core_gold_max_abs_diff=ch["core_gold_max_abs_diff"])
    M("core.gold_and_dense_s", time.time() - t0)
    arrs = {"gold": tok, "core_top_ids": dense["core_ids"], "core_top_p": dense["core_p"], "kn_top_ids": dense["base_ids"],
            "kn_top_p": dense["base_p"], "core_p_gold": rec.p_gold, "bucket": rec.bucket, "kstar_all": rec.kstar,
            "ptr_len": rec.feats.ptr_len, "ptr_cand": rec.feats.ptr_cand}
    del core, dense
    gc.collect()
    # literals and token tables exactly as in arm (i)
    cache_dir = ROOT / cfg["cache_run"] / "cache"
    hp_gold = C.load_arrays(cache_dir, "hprime", names=["gold"])["gold"]
    km = np.load(split_dir(protocol.CORPUS, V, "train") / "kind_map.npy")
    tabs = crtm.token_tables(km, train_counts_without_hprime(hp_gold), ROOT / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json")
    lsets = crtm.literal_sets(crtm.build_literals(tabs["c2_used"]), int(tabs["idx4"].max()))
    gates = load_gates(arm1_run, lsets)
    M("gates", sorted(gates), label="unverified")
    n = tok.shape[0]
    lp = {"core": np.log(rec.p_gold)}
    for g in gates:
        lp[g] = np.empty(n)
    t0 = time.time()
    chunk = 8192
    for lo in range(0, n, chunk):
        pos = np.arange(lo, min(n, lo + chunk), dtype=np.int64)
        pa = crtm.gather(arrs, pos, kstar_name="kstar_all")
        gs = crtm.gold_slots(pa.top_ids, pa.gold)
        f, _, _ = crtm.pair_features(tok, ds, pa, np.broadcast_to(np.arange(crtm.TOPK), (pos.shape[0], crtm.TOPK)), tabs)
        rd = crtm.readout_data(pa, gs)
        for g, G in gates.items():
            lp[g][lo:lo + pos.shape[0]] = crtm.apply_readout(G["score"](f), rd, G["fit"])
        if (lo // chunk) % 20 == 0:
            log(f"  scored {lo + pos.shape[0]}/{n} positions ({time.time() - t0:.0f} s)")
    M("blimp.scoring_s", time.time() - t0)
    doc_of = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
    sent = {g: np.bincount(doc_of, weights=v, minlength=ds.shape[0] - 1) for g, v in lp.items()}
    correct = {g: (s_[0::2] > s_[1::2]) for g, s_ in sent.items()}
    acc = {g: float(100.0 * c.mean()) for g, c in correct.items()}
    for g, a in acc.items():
        M(f"blimp_dev.{g}.accuracy_points", a, label="measured")
    uid = rows["UID"][di]
    per_par = {u: {g: float(100.0 * correct[g][uid == u].mean()) for g in correct} for u in np.unique(uid)}
    M("blimp_dev.per_paradigm_accuracy_points", per_par, label="measured")
    ctrls = [g for g in ("LR", "LR2", "HGB") if g in correct]
    best_ctrl = max(ctrls, key=lambda g: acc[g])
    M("arm2.best_control", best_ctrl, label="derived",
      note="the control with the highest dev accuracy (conservative: TM must clear every control); the control chosen on 2a in arm (i) is reported too")
    out = {}
    pairs_ = [("TM", "core"), ("TM", best_ctrl), ("TM", "C0"), ("LR", "core"), ("LR2", "core"), ("HGB", "core"),
              ("C0", "core"), ("TM_norel", "core"), ("TM_ctxid", "core"), ("TM", "TM_norel"), ("TM", "TM_ctxid")]
    pairs_ += [("TM", c) for c in ctrls if c != best_ctrl]
    for a, b in pairs_:
        if a in correct and b in correct:
            d = crtm.paired_bootstrap_points(correct[a], correct[b])
            out[(a, b)] = d
            M(f"blimp_dev.contrast.{a}_minus_{b}", d, label="derived",
              note="accuracy points, a minus b; paired bootstrap over the dev pairs, 2,000 resamples")
    v2 = crtm.verdict_arm2(out[("TM", "core")], out[("TM", best_ctrl)])
    M("verdict.arm_ii", v2, label="derived", best_control=best_ctrl)
    # arm (i) verdict from the arm1 run, then the P1 verdict
    v1 = None
    mpath = arm1_run / "metrics.json"
    if mpath.exists():
        mj = json.loads(mpath.read_text())
        if "verdict.arm_i" in mj:
            v1 = mj["verdict.arm_i"]["value"]
            b1 = mj.get("arm1.best_control_chosen_on_2a", {}).get("value")
            if b1 in correct:
                M("blimp_dev.contrast.TM_minus_arm1_chosen_control", crtm.paired_bootstrap_points(correct["TM"], correct[b1]),
                  label="derived", control=b1)
    vp = crtm.verdict_p1(v1, v2)
    M("verdict.P1", vp, label="derived", arm_i_source=str(mpath))
    np.savez_compressed(run / "blimp_dev_sentence_logprobs.npz", uid=uid, pairID=rows["pair_id"][di],
                        **{f"lp_{g}": v for g, v in sent.items()})
    M("wall_s", time.time() - T0)
    M("failed_checks", FAILED, label="measured")
    return 1 if FAILED else 0


# ------------------------------------------------------------------ main
def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["arm1", "blimp-hashes", "arm2"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True, help="scratch directory for the pair stores (outside results/)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--arm1-run", default=None)
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    cfg = json.loads(Path(args.config).read_text())
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if args.mode == "arm1":
        return arm1(args, cfg)
    if args.mode == "blimp-hashes":
        return blimp_hashes(args, cfg)
    if not args.arm1_run:
        raise SystemExit("arm2 needs --arm1-run")
    return arm2(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
