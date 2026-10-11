"""Lane B1: the TA-learned copy selector over the frozen CVTM-core at full data (docs/13 amendment A4,
H9.1 and H9.2, and gate G1 T2). Model, literals, gates and read-out: `src/tmlm/cvtm/copy_selector.py`.
VALIDATION ONLY: the test split is never loaded.

Protocol (docs/13 A4; frozen in `experiments/cvtm/configs/cvtm_copy_selector_*.json` before any
confirmatory run):
- Base: the frozen CVTM-core of `results/20260924T085728Z_cvtm_cache_build/` (lane L0). On the held-out
  training documents H' the core is the frozen EM weights over the KN-5 fitted WITHOUT H' (the T2
  inputs of A4, cache group "hprime"); on validation it is the frozen core itself.
- H' residue 1 documents ("t2_train") give the training pairs: every positive pair of every residue-1
  position, and a Bernoulli(0.01) pool of its negative pairs (keyed by stable document ID, offset and
  candidate). Each seed draws its training set from these: 250,000 positives and 750,000 negatives.
- H' residue 2 documents ("selection") give the selection positions S2: 1 in 8 of their positions,
  chosen by mix64 of (stable document ID, offset). Every configuration of every gate is scored on
  S2 with its own pooled (beta, a) fitted on S2; the chosen configuration then gets the per-bucket
  (beta_b, a_b) fitted on S2. Equal selection budgets: 8 configurations per gate.
- Validation: every position, bits per byte (protocol CP), document-block bootstrap (60 documents,
  2,000 resamples, the BOOT_SEED of tmlm.cvtm.protocol), labelled derived.

Modes:
- build: the pair stores (per-position arrays, pair features, P_core of every candidate, packed
  literal columns) of residue 1 (pool), S2 and validation, with pipeline checks against the cache
  (gold entries and top-64 of the core rows bit for bit). One logged run, 3 CPU slots.
- seed: one confirmatory seed (1, 2 or 3) or an exploratory pilot (seed >= 100): sample, train and
  select every gate, fit the read-out, score validation, readable TM clauses. One CPU slot.
- verdicts: H9.1, H9.2 and G1 T2 from the three seed runs.

Launch (see the configs for the exact lines):

    scripts/cpu_slot.sh 3 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py \\
        --no-gpu-sampler --name cvtm_copy_selector_pairs --config experiments/cvtm/configs/cvtm_copy_selector_pairs.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_copy_selector.py build --config experiments/cvtm/configs/cvtm_copy_selector_pairs.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.cvtm import cache as C  # noqa: E402
from tmlm.cvtm import copy_selector as CS  # noqa: E402
from tmlm.cvtm import protocol  # noqa: E402

LN2 = math.log(2.0)
GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False
V = protocol.VOCAB


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[copy_selector {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


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
    METRICS[name] = {"value": _plain(value), "label": label, **_plain(extra)}
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_metrics({name: METRICS[name]}, on_conflict="replace")
    if isinstance(value, (int, float, str, bool, np.floating, np.integer)) or value is None:
        log(f"{name} = {value}")


def check(name: str, ok: bool, **details) -> None:
    M(f"check.{name}", bool(ok), details=details)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {details}")


def out_dir(args) -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(args.out or os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_copy_selector_unlogged"
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    """The memory rule: start a step only when memory.current + its peak < 0.80 memory.max (poll 2 minutes)."""
    waited = 0
    while True:
        cur, ws, mx = cgroup()
        if cur + peak_gib * GIB < 0.80 * mx:
            M(f"memory_rule.{tag}", {"current_gib": cur / GIB, "working_set_gib": ws / GIB, "max_gib": mx / GIB,
                                     "peak_gib": peak_gib, "waited_s": waited}, label="measured")
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory rule for {tag} not satisfied after {waited} s")
        log(f"memory rule for {tag}: current {cur / GIB:.1f} GiB + {peak_gib} GiB >= 0.8 x {mx / GIB:.1f} GiB, waiting")
        time.sleep(poll_s)
        waited += poll_s


def load_cfg(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


# ------------------------------------------------------------------ build: parallel pair pass
_ST: dict = {}


def _unit(args: tuple[int, int]) -> dict:
    """Process positions[lo:hi] of the current group in chunks (runs in a forked worker)."""
    lo, hi = args
    st = _ST
    core, rec, tabs, corder = st["core"], st["rec"], st["tabs"], st["corder"]
    positions = st["positions"][lo:hi]
    mode, chunk = st["mode"], st["chunk"]
    top_ids, top_p = st["top_ids"], st["top_p"]
    res: dict = {"lo": lo, "hi": hi, "gold_mismatch": 0, "top_mismatch": 0, "row_sum_max_abs_err": 0.0,
                 "ncand_mismatch": 0, "feat_max": np.zeros(CS.NF, np.int64), "pcand_sum_max_abs_err": 0.0}
    ncand = np.zeros(hi - lo, np.int64)
    pstat = np.zeros((hi - lo, 7))
    if mode == "store":
        mm = {k: np.load(Path(st["gdir"]) / f"{k}.npy", mmap_mode="r+") for k in ("feat", "cand", "lab", "pc")}
        offs = st["offs"]
    else:
        pf, pp, nf, npos, ncd = [], [], [], [], []
    for a in range(0, hi - lo, chunk):
        b = min(hi - lo, a + chunk)
        pos = positions[a:b]
        ch = CS.pairs_for_positions(core, rec, pos, tabs, corder, CS.CAP, topk=64)
        ncand[a:b] = ch.ncand
        pstat[a:b] = ch.pstat
        res["gold_mismatch"] += int(np.sum(ch.pstat[:, 0] != rec.p_gold[pos]))
        ids, pv = ch.core_top
        res["top_mismatch"] += int(np.sum(np.any(ids != np.asarray(top_ids[pos]).astype(np.int64), axis=1)
                                          | np.any(pv != np.asarray(top_p[pos]), axis=1)))
        res["row_sum_max_abs_err"] = max(res["row_sum_max_abs_err"], float(np.max(np.abs(ch.pstat[:, 3] - 1.0))))
        res["ncand_mismatch"] += int(np.sum(ch.ncand != st["ncand_pre"][lo + a: lo + b]))
        if ch.feat.shape[0]:
            res["feat_max"] = np.maximum(res["feat_max"], ch.feat.max(axis=0).astype(np.int64))
        if mode == "store":
            p0, p1 = int(offs[lo + a]), int(offs[lo + b])
            if p1 - p0 != ch.feat.shape[0]:
                raise RuntimeError("pair count differs from the precomputed offsets")
            mm["feat"][p0:p1] = ch.feat
            mm["cand"][p0:p1] = ch.cand
            mm["lab"][p0:p1] = ch.lab
            mm["pc"][p0:p1] = ch.pcand
        else:
            rep = np.repeat(pos, ch.ncand)
            isp = ch.lab == 1
            pf.append(ch.feat[isp])
            pp.append(rep[isp])
            sel = CS.pool_select(ch, st["docid_pos"][pos], st["off_pos"][pos])
            nf.append(ch.feat[sel])
            npos.append(rep[sel])
            ncd.append(ch.cand[sel])
            res["n_neg_total"] = res.get("n_neg_total", 0) + int((ch.lab == 0).sum())
    if mode == "store":
        for m in mm.values():
            m.flush()
        del mm
    else:
        res["pos_feat"] = np.concatenate(pf) if pf else np.zeros((0, CS.NF), np.uint8)
        res["pos_pos"] = np.concatenate(pp) if pp else np.zeros(0, np.int64)
        res["neg_feat"] = np.concatenate(nf) if nf else np.zeros((0, CS.NF), np.uint8)
        res["neg_pos"] = np.concatenate(npos) if npos else np.zeros(0, np.int64)
        res["neg_cand"] = np.concatenate(ncd) if ncd else np.zeros(0, np.uint16)
    res["ncand"] = ncand
    res["pstat"] = pstat
    return res


def run_group(name: str, mode: str, core, rec, tabs, corder, positions, gdir: Path, top_ids, top_p, docid_pos,
              off_pos, workers: int, chunk: int) -> dict:
    t0 = time.time()
    positions = np.asarray(positions, np.int64)
    ncand_pre, ndist_pre = CS.candidate_counts(rec.tok, rec.doc_start, positions, CS.CAP)
    n_pairs = int(ncand_pre.sum())
    log(f"group {name}: {positions.shape[0]} positions, {n_pairs} pairs ({mode})")
    offs = np.zeros(positions.shape[0] + 1, np.int64)
    offs[1:] = np.cumsum(ncand_pre)
    gdir.mkdir(parents=True, exist_ok=True)
    if mode == "store":
        for k, dt, shp in (("feat", np.uint8, (n_pairs, CS.NF)), ("cand", np.uint16, (n_pairs,)),
                           ("lab", np.uint8, (n_pairs,)), ("pc", np.float64, (n_pairs,))):
            m = np.lib.format.open_memmap(gdir / f"{k}.npy", mode="w+", dtype=dt, shape=shp)
            del m
    # work units: contiguous position ranges of about equal cost (history length + a constant)
    cost = (positions - rec.doc_start[np.searchsorted(rec.doc_start, positions, side="right") - 1]).astype(np.float64) * 0.02 + 1.0
    cum = np.cumsum(cost)
    n_units = max(1, min(positions.shape[0], workers * 16))
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n_units) / n_units)
    bounds = np.unique(np.concatenate([[0], cuts, [positions.shape[0]]]))
    units = [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.shape[0] - 1) if bounds[i + 1] > bounds[i]]
    _ST.update(core=core, rec=rec, tabs=tabs, corder=corder, positions=positions, mode=mode, chunk=chunk,
               top_ids=top_ids, top_p=top_p, gdir=str(gdir), offs=offs, ncand_pre=ncand_pre, docid_pos=docid_pos,
               off_pos=off_pos)
    results = []
    try:
        if workers <= 1:
            for i, u in enumerate(units):
                results.append(_unit(u))
                if i % 8 == 0:
                    log(f"  {name}: unit {i + 1}/{len(units)} ({time.time() - t0:.0f} s)")
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(int(workers)) as pool:
                for i, r in enumerate(pool.imap(_unit, units)):
                    results.append(r)
                    if i % 8 == 0:
                        log(f"  {name}: unit {i + 1}/{len(units)} ({time.time() - t0:.0f} s)")
    finally:
        _ST.clear()
    ncand = np.concatenate([r["ncand"] for r in results])
    pstat = np.concatenate([r["pstat"] for r in results])
    out = {"positions": positions, "offs": offs, "ncand": ncand, "pstat": pstat, "ndist": ndist_pre,
           "n_pairs": n_pairs, "wall_s": time.time() - t0, "units": len(units)}
    for k in ("gold_mismatch", "top_mismatch", "ncand_mismatch"):
        out[k] = int(sum(r[k] for r in results))
    out["row_sum_max_abs_err"] = float(max(r["row_sum_max_abs_err"] for r in results))
    out["feat_max"] = np.max(np.stack([r["feat_max"] for r in results]), axis=0)
    if mode == "pool":
        for k in ("pos_feat", "pos_pos", "neg_feat", "neg_pos", "neg_cand"):
            out[k] = np.concatenate([r[k] for r in results])
        out["n_neg_total"] = int(sum(r.get("n_neg_total", 0) for r in results))
    return out


def coverage(name: str, g: dict) -> dict:
    ps = g["pstat"]
    gh, gc_ = ps[:, 1] > 0, ps[:, 2] > 0
    cov = {"positions": int(ps.shape[0]), "gold_in_document_history": float(gh.mean()),
           "gold_among_candidates": float(gc_.mean()), "lost_by_cap": float((gh & ~gc_).mean()),
           "mean_candidates": float(g["ncand"].mean()), "share_positions_at_cap": float((g["ncand"] >= CS.CAP).mean()),
           "mean_distinct_history_uncapped": float(g["ndist"].mean()), "pairs": int(g["n_pairs"])}
    M(f"coverage.{name}", cov, note=f"candidates = the {CS.CAP} most recent distinct earlier tokens of the document")
    return cov


def group_checks(name: str, g: dict) -> None:
    check(f"{name}.core_row_gold_equals_cache_bit_for_bit", g["gold_mismatch"] == 0, mismatches=g["gold_mismatch"])
    check(f"{name}.core_top64_equals_cache_bit_for_bit", g["top_mismatch"] == 0, mismatches=g["top_mismatch"])
    check(f"{name}.core_row_sums_to_one", g["row_sum_max_abs_err"] <= 1e-12, max_abs_err=g["row_sum_max_abs_err"])
    check(f"{name}.candidate_counts_equal_precount", g["ncand_mismatch"] == 0, mismatches=g["ncand_mismatch"])
    fm = np.asarray(g["feat_max"])
    check(f"{name}.features_within_bounds", bool(np.all(fm <= CS.FEAT_MAX)),
          feat_max=dict(zip(CS.FEATS, fm.tolist())))


def save_store_meta(gdir: Path, name: str, g: dict, rec, bucket, lpgold, doc_local) -> dict:
    """Per-position arrays of a store group, the packed literal columns and label bits."""
    pos = g["positions"]
    ps = g["pstat"]
    np.save(gdir / "positions.npy", pos)
    np.save(gdir / "offs.npy", g["offs"])
    np.save(gdir / "gslot.npy", ps[:, 6].astype(np.int32))
    np.save(gdir / "ghist.npy", (ps[:, 1] > 0).astype(np.uint8))
    np.save(gdir / "lpgold.npy", lpgold)
    np.save(gdir / "bucket.npy", np.asarray(bucket).astype(np.int64))
    np.save(gdir / "doc.npy", np.asarray(doc_local).astype(np.int32))
    np.save(gdir / "ndist.npy", g["ndist"].astype(np.int32))
    return {"positions": int(pos.shape[0]), "pairs": int(g["n_pairs"])}


def pack_store(gdir: Path, lits, blk_pairs: int = 64 * 65536) -> dict:
    t0 = time.time()
    feat = np.load(gdir / "feat.npy", mmap_mode="r")
    lab = np.load(gdir / "lab.npy", mmap_mode="r")
    n = feat.shape[0]
    nb = (n + 63) // 64
    cols = np.lib.format.open_memmap(gdir / "cols.npy", mode="w+", dtype=np.uint64, shape=(nb, lits.n))
    labbits = np.zeros(nb, np.uint64)
    for a in range(0, n, blk_pairs):
        b = min(n, a + blk_pairs)
        cols[a // 64: (b + 63) // 64] = CS.pack_literals(np.asarray(feat[a:b]), lits)
        labbits[a // 64: (b + 63) // 64] = CS.pack_labels(np.asarray(lab[a:b]))
    cols.flush()
    del cols
    np.save(gdir / "labbits.npy", labbits)
    return {"blocks": int(nb), "pack_s": time.time() - t0}


def mode_build(cfg: dict, args) -> int:
    from tmlm.cvtm.model import CVTMCore, suffix_gold
    from tmlm.cvtm.suffix import SuffixClauses
    from tmlm.data.corpus_store import load_split

    b = cfg["build"]
    cd = ROOT / cfg["cache_dir"]
    od = out_dir(args)
    pdir = od / "pairs"
    pdir.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or b["workers"])
    chunk = int(b["chunk_positions"])
    M("config.params", cfg, label="unverified", note="the run's config (also config.json)")
    # ---------------- cache identity
    meta_sha = hashing.sha256_file(cd / C.META_FILE)
    check("cache_meta_sha256_as_frozen", meta_sha == cfg["cache_meta_sha256"], got=meta_sha, want=cfg["cache_meta_sha256"])
    meta = C.read_meta(cd)
    hashes = protocol.dataset_hashes(("train", "validation"), protocol.CORPUS, V)
    hashes["cvtm_cache_meta_json"] = meta_sha
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_dataset_hashes(hashes, on_conflict="replace")
    M("dataset_hashes", hashes, note="stores' meta.json digests as results/20260923T042350Z_CP_anchor/ plus document starts, "
                                     "per-document bytes and the L0 cache_meta.json")
    lits = CS.literal_construction()
    M("literals", {"n": lits.n, "n_reflective": int(lits.reflective.sum()), "names": lits.names},
      note="imported from experiments/cvtm_early_logical_attention.py (literal_spec, literal_tables); B renamed core")
    # ---------------- H' arrays and token tables
    A = C.load_arrays(cd, "hprime", mmap=True)
    tok = np.asarray(A["gold"]).astype(np.int64)
    ds = np.asarray(A["doc_start"]).astype(np.int64)
    doc_index = np.asarray(A["doc_index"]).astype(np.int64)
    residue = np.asarray(A["residue"]).astype(np.int64)
    n_hp = tok.shape[0]
    doc_of = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
    off_pos = np.arange(n_hp) - ds[doc_of]
    docid_pos = doc_index[doc_of]
    res_pos = residue[doc_of]
    tr = load_split(protocol.CORPUS, V, "train")
    counts_all = np.bincount(np.asarray(tr.tok).astype(np.int64), minlength=V)
    counts_hp = np.bincount(tok, minlength=V)
    tabs = CS.token_tables(counts_all[:V] - counts_hp[:V], np.asarray(tr.kind_map),
                           ROOT / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json", V)
    del tr
    M("token_tables", {"fcls_counts_from": "training split without H' (residues 0, 1, 2)",
                       "clusters": "kind_map of the V 8000 train store (levels 1, 2, 3 = 16, 256, 4096 clusters)",
                       "type_bits": {CS.exploratory_module().TYPE_BITS[k]: int(((tabs['ctype'] >> k) & 1).sum()) for k in range(6)}})
    pos_r1 = np.flatnonzero(res_pos == 1)
    pos_s2 = np.flatnonzero((res_pos == 2) & CS.s2_mask(docid_pos, off_pos))
    if args.smoke_docs:
        k = int(args.smoke_docs)
        keep_docs = np.concatenate([np.flatnonzero(residue == r)[:k] for r in (1, 2)])
        keep = np.isin(doc_of, keep_docs)
        pos_r1 = pos_r1[keep[pos_r1]]
        pos_s2 = pos_s2[keep[pos_s2]]
    half = CS.s2_half(docid_pos[pos_s2], off_pos[pos_s2])
    pos_s2a, pos_s2b = pos_s2[half == 0], pos_s2[half == 1]
    M("groups.hprime", {"residue1_positions": int(pos_r1.shape[0]), "s2_positions": int(pos_s2.shape[0]),
                        "s2a_positions": int(pos_s2a.shape[0]), "s2b_positions": int(pos_s2b.shape[0]),
                        "s2_halves": "bit 3 of the same hash: S2a (0) scores every configuration, S2a + S2b fit the chosen read-out",
                        "residue2_positions": int((res_pos == 2).sum()), "s2_fraction_of_residue2": float(pos_s2.shape[0] / max(1, (res_pos == 2).sum())),
                        "s2_rule": f"mix64(mix64(doc_id ^ {hex(CS.S2_SALT)}) + offset) mod {CS.S2_DEN} == 0"})
    # ---------------- the H' core: frozen EM weights over the KN without H'
    wait_for_memory(float(b["peak_gib_hprime"]), "hprime_core")
    t0 = time.time()
    kn = C.load_kn_variant(cd, "noHp", verify=True)
    sc = SuffixClauses.from_kn(kn, backend="numpy")
    del kn
    gc.collect()
    with np.load(cd / C.CORE_FILE, allow_pickle=False) as z:
        weights = np.array(z["weights"])
        cmeta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    ccfg = dict(cmeta["config"])
    core_hp = CVTMCore(suffix=sc, weights=weights, config=ccfg, fit_inputs=None)
    M("hprime.core_load_s", time.time() - t0)
    lnp, ks = suffix_gold(sc, tok, ds)
    check("hprime.p_kn_noHp_equals_cache", bool(np.array_equal(np.exp(lnp), np.asarray(A["p_kn_noHp"]))))
    check("hprime.kstar_noHp_equals_cache", bool(np.array_equal(ks.astype(np.int8), np.asarray(A["kstar_noHp"]))))
    feats = C.doc_features_of(A)
    rec = core_hp.gold(tok, ds, lnp_s=lnp, kstar=ks, feats=feats)
    check("hprime.core_p_gold_equals_cache_bit_for_bit", bool(np.array_equal(rec.p_gold, np.asarray(A["core_p_gold"]))))
    check("hprime.bucket_equals_cache", bool(np.array_equal(rec.bucket.astype(np.int16), np.asarray(A["bucket"]))))
    corder = CS.cache_order(np.asarray(A["doc_n"]))
    # warm the numba kernels before forking
    CS.pairs_for_positions(core_hp, rec, pos_s2[:2], tabs, corder, CS.CAP, topk=64)
    # ---------------- residue 1: training pool
    g1 = run_group("r1", "pool", core_hp, rec, tabs, corder, pos_r1, pdir / "r1", A["core_top_ids"], A["core_top_p"],
                   docid_pos, off_pos, workers, chunk)
    group_checks("r1", g1)
    coverage("r1", g1)
    d1 = pdir / "r1"
    for k in ("pos_feat", "pos_pos", "neg_feat", "neg_pos", "neg_cand"):
        np.save(d1 / f"{k}.npy", g1[k])
    np.save(d1 / "positions.npy", g1["positions"])
    np.save(d1 / "ncand.npy", g1["ncand"])
    M("r1.pool", {"positives": int(g1["pos_feat"].shape[0]), "pooled_negatives": int(g1["neg_feat"].shape[0]),
                  "negatives_total": int(g1["n_neg_total"]), "pool_rate": CS.POOL_RATE,
                  "pool_rate_realised": float(g1["neg_feat"].shape[0] / max(1, g1["n_neg_total"])), "wall_s": g1["wall_s"]})
    del g1
    gc.collect()
    # ---------------- S2 (two halves)
    for gname, pos_g in (("s2a", pos_s2a), ("s2b", pos_s2b)):
        g2 = run_group(gname, "store", core_hp, rec, tabs, corder, pos_g, pdir / gname, A["core_top_ids"], A["core_top_p"],
                       docid_pos, off_pos, workers, chunk)
        group_checks(gname, g2)
        coverage(gname, g2)
        info2 = save_store_meta(pdir / gname, gname, g2, rec, np.asarray(A["bucket"])[pos_g],
                                np.log(np.asarray(A["core_p_gold"])[pos_g]), doc_of[pos_g])
        info2.update(pack_store(pdir / gname, lits))
        M(f"{gname}.store", {**info2, "wall_s": g2["wall_s"]})
        del g2
        gc.collect()
    del core_hp, rec, sc, feats, lnp, ks
    gc.collect()
    # ---------------- validation: the frozen core
    wait_for_memory(float(b["peak_gib_validation"]), "validation_core")
    t0 = time.time()
    core = C.load_core(cd, backend="numpy")
    M("validation.core_load_s", time.time() - t0)
    check("validation.core_state_hash_as_frozen", core.state_hash() == cfg["core_state_hash"], got=core.state_hash())
    ev = protocol.load_eval_split("validation")
    Av = C.load_arrays(cd, "validation", mmap=True)
    check("validation.tokens_equal_cache", bool(np.array_equal(np.asarray(ev.tok).astype(np.int64), np.asarray(Av["gold"]).astype(np.int64))))
    fv = C.doc_features_of(Av)
    recv = core.gold(ev.tok, ev.doc_start, feats=fv)
    check("validation.core_p_gold_equals_cache_bit_for_bit", bool(np.array_equal(recv.p_gold, np.asarray(Av["core_p_gold"]))))
    check("validation.bucket_equals_cache", bool(np.array_equal(recv.bucket.astype(np.int16), np.asarray(Av["bucket"]))))
    core_bpb = float(-np.log2(recv.p_gold).sum() / ev.n_bytes)
    check("validation.core_bpb_equals_L0", abs(core_bpb - float(cfg["core_validation_bpb"])) <= 1e-12 or bool(args.smoke_docs), got=core_bpb,
          want=cfg["core_validation_bpb"])
    M("validation.core_bits_per_byte", core_bpb, note="CVTM-core, frozen (L0 run), recomputed from the cache core_p_gold")
    corder_v = CS.cache_order(np.asarray(Av["doc_n"]))
    pos_v = np.arange(ev.tok.shape[0], dtype=np.int64)
    if args.smoke_docs:
        pos_v = pos_v[: int(ev.doc_start[min(int(args.smoke_docs), ev.n_docs)])]
    docv = np.repeat(np.arange(ev.n_docs), np.diff(ev.doc_start))
    CS.pairs_for_positions(core, recv, pos_v[:2], tabs, corder_v, CS.CAP, topk=64)
    gv = run_group("val", "store", core, recv, tabs, corder_v, pos_v, pdir / "val", Av["core_top_ids"], Av["core_top_p"],
                   docv, np.arange(ev.tok.shape[0], dtype=np.int64) - ev.doc_start[docv], workers, chunk)
    group_checks("val", gv)
    coverage("val", gv)
    infov = save_store_meta(pdir / "val", "val", gv, recv, np.asarray(Av["bucket"])[pos_v],
                            np.log(np.asarray(Av["core_p_gold"])[pos_v]), docv[pos_v])
    np.save(pdir / "val" / "doc_bytes.npy", ev.doc_bytes)
    infov.update(pack_store(pdir / "val", lits))
    M("val.store", {**infov, "wall_s": gv["wall_s"], "n_bytes": int(ev.n_bytes), "n_docs": int(ev.n_docs)})
    # candidate mass check: the candidates' P_core sums to at most the row (so Z > 0 for every tilt)
    for gname in ("s2a", "s2b", "val"):
        st = Store(pdir / gname)
        pcm = 0.0
        for a, bnd in st.pos_chunks():
            seg = np.asarray(st.pc[st.offs[a]:st.offs[bnd]])
            cs_ = np.concatenate([[0.0], np.cumsum(seg)])
            sums = cs_[st.offs[a + 1:bnd + 1] - st.offs[a]] - cs_[st.offs[a:bnd] - st.offs[a]]
            pcm = max(pcm, float(sums.max()) if sums.shape[0] else 0.0)
        check(f"{gname}.candidate_mass_at_most_one", pcm <= 1.0 + 1e-12, max_candidate_mass=pcm)
    store_meta = {"format": "tmlm.cvtm.copy_selector.pairs.v1", "cap": CS.CAP, "feats": CS.FEATS,
                  "feat_max": CS.FEAT_MAX.tolist(), "s2_salt": CS.S2_SALT, "s2_den": CS.S2_DEN,
                  "pool_salt": CS.POOL_SALT, "pool_rate": CS.POOL_RATE, "cache_dir": cfg["cache_dir"],
                  "cache_meta_sha256": meta_sha, "core_state_hash": core.state_hash(), "n_literals": lits.n,
                  "validation_n_bytes": int(ev.n_bytes), "core_validation_bpb": core_bpb,
                  "smoke_docs": int(args.smoke_docs or 0)}
    (pdir / "store_meta.json").write_text(json.dumps(store_meta, indent=1, sort_keys=True))
    M("wall_s", time.time() - T0)
    return 1 if FAILED else 0


# ------------------------------------------------------------------ seed mode
class Store:
    """A memory-mapped pair store (S2 or validation) with its read-out inputs."""

    def __init__(self, gdir: Path, load_pc: bool = True) -> None:
        self.dir = gdir
        self.offs = np.load(gdir / "offs.npy")
        self.gslot = np.load(gdir / "gslot.npy").astype(np.int64)
        self.lpgold = np.load(gdir / "lpgold.npy")
        self.bucket = np.load(gdir / "bucket.npy").astype(np.int64)
        self.doc = np.load(gdir / "doc.npy").astype(np.int64)
        self.feat = np.load(gdir / "feat.npy", mmap_mode="r")
        self.cols = np.load(gdir / "cols.npy", mmap_mode="r")
        self.lab = np.load(gdir / "lab.npy", mmap_mode="r")
        self.labbits = np.load(gdir / "labbits.npy")
        self.pc = np.load(gdir / "pc.npy", mmap_mode="r")
        self.n_pos = int(self.offs.shape[0] - 1)
        self.n_pairs = int(self.offs[-1])

    def rd(self, a: int = 0, b: int | None = None) -> CS.ReadoutData:
        b = self.n_pos if b is None else b
        p0, p1 = int(self.offs[a]), int(self.offs[b])
        return CS.ReadoutData(offs=self.offs[a:b + 1] - p0, gslot=self.gslot[a:b], lpgold=self.lpgold[a:b],
                              bucket=self.bucket[a:b], pc=np.ascontiguousarray(self.pc[p0:p1]))

    def pos_chunks(self, max_pairs: int = 1 << 24):
        """Contiguous position ranges of at most about max_pairs pairs."""
        a = 0
        while a < self.n_pos:
            b = int(np.searchsorted(self.offs, self.offs[a] + max_pairs, side="right")) - 1
            b = max(a + 1, min(self.n_pos, b))
            yield a, b
            a = b


def concat_rd(a: "CS.ReadoutData", b: "CS.ReadoutData") -> "CS.ReadoutData":
    return CS.ReadoutData(offs=np.concatenate([a.offs, b.offs[1:] + a.offs[-1]]), gslot=np.concatenate([a.gslot, b.gslot]),
                          lpgold=np.concatenate([a.lpgold, b.lpgold]), bucket=np.concatenate([a.bucket, b.bucket]),
                          pc=np.concatenate([a.pc, b.pc]))


def score_store(gate: dict, st: Store, lits, a: int = 0, b: int | None = None, max_pairs: int = 1 << 23) -> np.ndarray:
    """Raw gate scores of the pairs of positions a..b-1 of a store (float64)."""
    b = st.n_pos if b is None else b
    p0, p1 = int(st.offs[a]), int(st.offs[b])
    out = np.empty(p1 - p0)
    for q0 in range(p0, p1, max_pairs):
        q1 = min(p1, q0 + max_pairs)
        kind = gate["kind"]
        if kind == "tm":
            out[q0 - p0:q1 - p0] = CS.tm_scores(st.cols, q0, q1, gate["clauses"])
        elif kind == "lr":
            out[q0 - p0:q1 - p0] = gate["model"].score(np.asarray(st.feat[q0:q1]), gate["theta"], gate["tables"])
        elif kind == "hgb":
            out[q0 - p0:q1 - p0] = CS.hgb_scores(np.asarray(st.feat[q0:q1]), lits, gate["ens"], [gate["stage"]])[0]
        else:
            raise ValueError(kind)
    return out


def hgb_scores_multi(ens, st: Store, lits, stages, max_pairs: int = 1 << 22) -> np.ndarray:
    out = np.empty((len(stages), st.n_pairs))
    for q0 in range(0, st.n_pairs, max_pairs):
        q1 = min(st.n_pairs, q0 + max_pairs)
        out[:, q0:q1] = CS.hgb_scores(np.asarray(st.feat[q0:q1]), lits, ens, stages)
    return out


def mode_seed(cfg: dict, args) -> int:
    from tmu.models.classification.vanilla_classifier import TMClassifier

    sd = cfg["seed_protocol"]
    seed = int(args.seed)
    exploratory = seed >= 100
    if not exploratory and seed not in cfg["seeds"]:
        raise SystemExit(f"seed {seed} is not a registered seed {cfg['seeds']}")
    if exploratory and not cfg.get("exploratory", False):
        raise SystemExit("seeds >= 100 run only under the exploratory pilot config")
    label = "measured"
    M("config.params", cfg, label="unverified")
    M("seed", seed, label="unverified", note="exploratory pilot" if exploratory else "confirmatory (docs/13 section 3)")
    pairs_run = ROOT / cfg["pairs_run"] if not args.pairs_dir else Path(args.pairs_dir).parent
    pdir = Path(args.pairs_dir) if args.pairs_dir else pairs_run / "pairs"
    smeta = json.loads((pdir / "store_meta.json").read_text())
    if not args.pairs_dir:
        rj = json.loads((pairs_run / "run.json").read_text())
        check("pairs_run_exit_0", rj.get("exit_status") == 0, exit_status=rj.get("exit_status"))
        mf = hashing.sha256_file(pairs_run / "MANIFEST.json")
        check("pairs_manifest_sha256_as_frozen", mf == cfg["pairs_manifest_sha256"], got=mf)
        hashes = protocol.dataset_hashes(("train", "validation"), protocol.CORPUS, V)
        hashes["cvtm_cache_meta_json"] = smeta["cache_meta_sha256"]
        hashes["copy_selector_pairs_manifest"] = mf
        if runlog.current_run_dir() is not None and not SMOKE:
            runlog.write_dataset_hashes(hashes, on_conflict="replace")
    wait_for_memory(float(sd.get("peak_gib", 6)), f"seed{seed}")
    lits = CS.literal_construction()
    L = lits.n
    # ---------------- training pairs of this seed
    d1 = pdir / "r1"
    pos_feat = np.load(d1 / "pos_feat.npy")
    neg_feat = np.load(d1 / "neg_feat.npy")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260924, 9101]))
    n_pos = min(int(sd["train_positives"]), pos_feat.shape[0])
    n_neg = min(int(sd["train_negatives"]), neg_feat.shape[0])
    ip = np.sort(rng.choice(pos_feat.shape[0], size=n_pos, replace=False))
    ineg = np.sort(rng.choice(neg_feat.shape[0], size=n_neg, replace=False))
    featT = np.concatenate([pos_feat[ip], neg_feat[ineg]])
    yT = np.concatenate([np.ones(n_pos, np.uint32), np.zeros(n_neg, np.uint32)])
    # C0: 1 percent of these pairs, stratified
    k_pos = max(1, int(round(sd["c0_fraction"] * n_pos)))
    k_neg = max(1, int(round(sd["c0_fraction"] * n_neg)))
    i0 = np.concatenate([np.sort(rng.choice(n_pos, size=k_pos, replace=False)),
                         n_pos + np.sort(rng.choice(n_neg, size=k_neg, replace=False))])
    M("train.pairs", {"positives": n_pos, "negatives": n_neg, "pool_positives": int(pos_feat.shape[0]),
                      "pool_negatives": int(neg_feat.shape[0]), "c0_pairs": int(i0.shape[0]),
                      "sampling": "uniform without replacement from all residue-1 positive pairs and from the Bernoulli(0.01) "
                                  "pool of residue-1 negative pairs, numpy SeedSequence([seed, 20260924, 9101])"})
    del pos_feat, neg_feat
    XT = lits.matrix(featT, np.uint32)
    X_c0 = np.ascontiguousarray(XT[i0])
    y_c0 = np.ascontiguousarray(yT[i0])
    s2 = Store(pdir / "s2a")                   # selection half: every configuration is scored here
    s2b = Store(pdir / "s2b")                  # second half: only the chosen configuration (read-out fit)
    val = Store(pdir / "val") if not args.no_validation else None
    rd2 = s2.rd()
    M("s2.size", {"s2a_positions": s2.n_pos, "s2a_pairs": s2.n_pairs, "s2b_positions": s2b.n_pos, "s2b_pairs": s2b.n_pairs,
                  "s2a_core_bits_per_position": float(-s2.lpgold.mean() / LN2)})
    n_buckets = int(cfg["n_buckets"])
    doc_bytes = None
    if val is not None:
        doc_bytes = np.load(pdir / "val" / "doc_bytes.npy").astype(np.float64)
        n_bytes = int(smeta["validation_n_bytes"])
        core_bits_pos = -val.lpgold / LN2
        core_bpb = float(core_bits_pos.sum() / n_bytes)
        if not smeta.get("smoke_docs"):
            check("validation.core_bpb_equals_L0", core_bpb == float(cfg["core_validation_bpb"]), got=core_bpb)
        M("row.core.validation_bits_per_byte", core_bpb, note="the frozen CVTM-core (L0 cache)")
    results: dict = {}
    per_tok: dict = {}
    budget = int(sd["configs_per_gate"])

    # ---------------- selection helpers
    def evaluate(tag: str, raw: np.ndarray) -> dict:
        mu = float(raw.mean())
        sdv = float(raw.std())
        sdv = sdv if sdv > 0 else 1.0
        z = raw                                  # standardised in place (the raw array is not used afterwards)
        z -= mu
        z /= sdv
        th, info = CS.fit_pooled(z, rd2)
        info.update({"mu": mu, "sigma": sdv})
        log(f"  {tag}: S2a {info['bits_per_position']:.5f} bits/position (core {info['core_bits_per_position']:.5f}), "
            f"beta {info['beta']:.4f} a {info['a']:.4f}")
        return {"z": z, "theta": th, "info": info}

    def finish_gate(name: str, sel: list, best: dict, gate: dict, t_gate: float, extra: dict | None = None) -> None:
        if len(sel) != budget:
            check(f"{name}.selection_budget_equals_{budget}", False, evaluated=len(sel))
        mu, sg = best["info"]["mu"], best["info"]["sigma"]
        zb = score_store(gate, s2b, lits)
        zb -= mu
        zb /= sg
        rdb = s2b.rd()
        # residue-2 criterion between gates (docs/13 A5 D0.2: the best control is chosen on residue 2 of H', never
        # on validation): bits per S2b position of the chosen configuration with the pooled (beta, a) fitted on
        # S2a. S2b took no part in choosing the configuration or in fitting that read-out (held out).
        th_a = np.asarray(best["theta"], np.float64)
        lpb = CS.readout_logp(zb, rdb, th_a[:1].copy(), th_a[1:].copy(), bucket=np.zeros(rdb.n, np.int64))
        s2b_crit = {"bits_per_position": float(-lpb.mean() / LN2), "core_bits_per_position": float(-rdb.lpgold.mean() / LN2),
                    "positions": int(rdb.n), "readout": "pooled (beta, a) fitted on S2a (the selection read-out)"}
        del lpb
        log(f"  {name}: S2b held-out {s2b_crit['bits_per_position']:.5f} bits/position "
            f"(core {s2b_crit['core_bits_per_position']:.5f})")
        rd_all = concat_rd(rd2, rdb)
        z_all = np.concatenate([best["z"], zb])
        del zb
        th_all, pinfo = CS.fit_pooled(z_all, rd_all, x0=best["theta"])
        be, al, binfo = CS.fit_buckets(z_all, rd_all, th_all, n_buckets, prior=float(sd["prior_positions"]))
        del z_all, rd_all
        gate.update({"mu": mu, "sigma": sg, "beta_b": be, "a_b": al, "theta0": th_all})
        rec = {"selection": sel, "chosen": best["cfg"], "pooled_s2a": best["info"], "s2b_heldout": s2b_crit,
               "pooled_s2": pinfo, "bucket_fit": binfo, "configs_evaluated_on_s2a": len(sel), **(extra or {})}
        if val is not None:
            t1 = time.time()
            lp = np.empty(val.n_pos)
            for a, b in val.pos_chunks():
                raw = score_store(gate, val, lits, a, b)
                z = (raw - gate["mu"]) / gate["sigma"]
                lp[a:b] = CS.readout_logp(z, val.rd(a, b), be, al)
            bits = -lp / LN2
            per_tok[name] = bits
            bpb = float(bits.sum() / n_bytes)
            rec["validation_bits_per_byte"] = bpb
            rec["validation_score_s"] = time.time() - t1
            db = np.bincount(val.doc, weights=bits, minlength=doc_bytes.shape[0])
            dc = np.bincount(val.doc, weights=core_bits_pos, minlength=doc_bytes.shape[0])
            bs = protocol.bootstrap_bpb(db, doc_bytes, BIDX, doc_bits_ref=dc)
            rec["validation_minus_core"] = bs
            M(f"row.{name}.validation_bits_per_byte", bpb, label=label)
            M(f"contrast.{name}_minus_core", bs, label="derived",
              note="bits per byte, document-block bootstrap (60 documents, 2,000 resamples, protocol.BOOT_SEED)")
        rec["gate_wall_s"] = time.time() - t_gate
        results[name] = rec
        M(f"gate.{name}", rec, label=label)
        gc.collect()

    if val is not None:
        BIDX = protocol.bootstrap_index(doc_bytes.shape[0])
        # read-out identity: zero tilt reproduces the core
        lp0 = np.empty(val.n_pos)
        for a, b in val.pos_chunks():
            rdv = val.rd(a, b)
            lp0[a:b] = CS.readout_logp(np.zeros(int(rdv.offs[-1])), rdv, np.zeros(n_buckets), np.zeros(n_buckets))
        dmax = float(np.max(np.abs(lp0 - val.lpgold)))
        check("readout_zero_tilt_equals_core", dmax <= 1e-12, max_abs_ln_diff=dmax)

    order = list(args.gates.split(",")) if args.gates else list(sd["gate_order"])
    # ---------------- TM and C0
    for gname in [g for g in order if g in ("TM", "C0")]:
        t_gate = time.time()
        if gname == "TM":
            X, y = XT, np.ascontiguousarray(yT)
        else:
            X, y = X_c0, y_c0
        sel, best = [], None
        for gi, (T, s) in enumerate(sd["tm_grid"]):
            tm = TMClassifier(int(sd["tm_clauses_per_class"]), T=int(T), s=float(s), platform="CPU",
                              weighted_clauses=True, seed=seed * 1000 + gi)
            for ep in range(1, int(sd["tm_epochs"]) + 1):
                t1 = time.time()
                tm.fit(X, y)
                fit_s = time.time() - t1
                cs = CS.tm_clause_set(tm, L)
                raw = score_store({"kind": "tm", "clauses": cs}, s2, lits)
                if gi == 0 and ep == 1:
                    # the bit-parallel evaluator against tmu's class sums on 3000 S2 pairs: the difference must be
                    # one constant (tmu's predict may count empty clauses; a constant cancels in the read-out)
                    rv = np.random.default_rng(7)
                    ii = np.sort(rv.choice(s2.n_pairs, size=min(3000, s2.n_pairs), replace=False))
                    Xc = lits.matrix(np.asarray(s2.feat[ii]), np.uint32)
                    _, csum = tm.predict(Xc, return_class_sums=True)
                    d = csum[:, 1].astype(np.int64) - csum[:, 0].astype(np.int64) - raw[ii].astype(np.int64)
                    ud = np.unique(d)
                    check(f"{gname}.tm_evaluator_equals_tmu_class_sum_difference",
                          bool(ud.shape[0] == 1 and int(ud[0]) in (0, cs.empty_const)),
                          distinct_offsets=ud.tolist()[:5], empty_clause_constant=cs.empty_const)
                ev_ = evaluate(f"{gname} T {T} s {s} epoch {ep}", raw)
                n_inc = np.diff(cs.ptr)
                cfg_rec = {"T": int(T), "s": float(s), "epoch": ep, "fit_s": fit_s, "nonempty_clauses": cs.n,
                           "mean_literals_per_nonempty_clause": float(n_inc.mean()) if cs.n else 0.0,
                           "s2a": ev_["info"]}
                sel.append(cfg_rec)
                if best is None or ev_["info"]["bits_per_position"] < best["info"]["bits_per_position"]:
                    best = {**ev_, "cfg": cfg_rec, "clauses": cs}
                del raw, ev_
            del tm
            gc.collect()
        gate = {"kind": "tm", "clauses": best["clauses"]}
        finish_gate(gname, sel, best, gate, t_gate, extra={
            "final_fit_capped": False,
            "iteration_cap": "not applicable: the epochs 1 and 2 are configurations, not a convergence cap"})
        if gname == "TM":
            tm_rules(best["clauses"], s2, val, lits, label)
            del X, XT
            XT = None
        del best, sel
    # ---------------- LR and LR2
    for gname in [g for g in order if g in ("LR", "LR2")]:
        t_gate = time.time()
        model = CS.TableLR(lits, pairwise=(gname == "LR2"))
        grid = sorted([float(x) for x in sd["lr_l2_grid" if gname == "LR" else "lr2_l2_grid"]], reverse=True)
        sel, best, theta = [], None, None
        maxit = int(sd["lr_maxiter"])
        for l2 in grid:
            theta_start = None if theta is None else theta.copy()
            theta, finfo = model.fit(featT, yT, l2, theta0=theta, maxiter=maxit)
            finfo["hit_iteration_cap"] = CS.lbfgs_hit_cap(finfo, maxit)
            T_ = model.tables(theta)
            raw = score_store({"kind": "lr", "model": model, "theta": theta, "tables": T_}, s2, lits)
            ev_ = evaluate(f"{gname} l2 {l2:g}", raw)
            cfg_rec = {"l2": l2, **finfo, "s2a": ev_["info"]}
            sel.append(cfg_rec)
            if best is None or ev_["info"]["bits_per_position"] < best["info"]["bits_per_position"]:
                best = {**ev_, "cfg": cfg_rec, "lr_theta": theta.copy(), "tables": T_, "theta_start": theta_start}
            del raw, ev_
        # docs/13 A5 D0.2: the chosen configuration's fit stopped at its iteration cap -> refit it once, from the
        # same starting point, with the cap raised four-fold; it replaces the capped fit (the selection is not
        # redone: the same configuration, better converged). Still capped -> reported, "beats" is not claimed.
        refit = {"needed": bool(best["cfg"]["hit_iteration_cap"]), "cap": maxit,
                 "capped_configurations": int(sum(1 for c in sel if c["hit_iteration_cap"]))}
        if refit["needed"]:
            l2b = float(best["cfg"]["l2"])
            th_r, fr = model.fit(featT, yT, l2b, theta0=best["theta_start"], maxiter=4 * maxit)
            fr["hit_iteration_cap"] = CS.lbfgs_hit_cap(fr, 4 * maxit)
            T_r = model.tables(th_r)
            raw = score_store({"kind": "lr", "model": model, "theta": th_r, "tables": T_r}, s2, lits)
            ev_r = evaluate(f"{gname} l2 {l2b:g} refit (cap {4 * maxit})", raw)
            refit.update({"refit_cap": 4 * maxit, "refit_fit": fr, "refit_s2a": ev_r["info"],
                          "s2a_bits_per_position_before": best["info"]["bits_per_position"]})
            best = {**ev_r, "cfg": {**best["cfg"], "refit": fr}, "lr_theta": th_r.copy(), "tables": T_r,
                    "theta_start": best["theta_start"]}
            del raw, ev_r
        refit["final_fit_capped"] = bool(best["cfg"]["refit"]["hit_iteration_cap"]) if refit["needed"] else False
        M(f"{gname}.iteration_cap_refit", refit, label=label,
          note="docs/13 A5 D0.2; the refit is the same configuration, so the selection budget is unchanged")
        if gname == "LR2":
            # table evaluation against the explicit quadratic form on 1000 training pairs
            W = model.W_of(best["lr_theta"])
            Xs = lits.matrix(featT[:1000], np.float64)
            explicit = np.einsum("ni,ij,nj->n", Xs, np.triu(W), Xs) + best["lr_theta"][-1]
            got = model.score(featT[:1000], best["lr_theta"])
            check("lr2_tables_equal_explicit_quadratic_form", float(np.max(np.abs(explicit - got))) <= 1e-9,
                  max_abs_diff=float(np.max(np.abs(explicit - got))))
        gate = {"kind": "lr", "model": model, "theta": best["lr_theta"], "tables": best["tables"]}
        finish_gate(gname, sel, best, gate, t_gate, extra={"iteration_cap_refit": refit,
                                                          "final_fit_capped": refit["final_fit_capped"]})
        del best, sel
    # ---------------- HGB
    if "HGB" in order:
        from sklearn.ensemble import HistGradientBoostingClassifier

        t_gate = time.time()
        XH = lits.matrix(featT, np.uint8)
        sel, best = [], None
        for gi, hp in enumerate(sd["hgb_grid"]):
            t1 = time.time()
            mdl = HistGradientBoostingClassifier(max_iter=int(max(hp["stages"])), learning_rate=float(hp["learning_rate"]),
                                                 max_leaf_nodes=int(hp["max_leaf_nodes"]), min_samples_leaf=20,
                                                 l2_regularization=0.0, early_stopping=False,
                                                 random_state=seed * 1000 + gi)
            mdl.fit(XH, yT)
            fit_s = time.time() - t1
            ens = CS.hgb_ensemble(mdl)
            ii = np.arange(min(2000, s2.n_pairs))
            ref = mdl.decision_function(lits.matrix(np.asarray(s2.feat[ii]), np.uint8))
            mine = CS.hgb_scores(np.asarray(s2.feat[ii]), lits, ens, [ens.n_trees])[0]
            check(f"hgb_evaluator_equals_sklearn.{gi}", bool(np.array_equal(ref, mine)),
                  max_abs_diff=float(np.max(np.abs(ref - mine))))
            stages = sorted(int(x) for x in hp["stages"])
            raws = hgb_scores_multi(ens, s2, lits, stages)
            for k, stg in enumerate(stages):
                ev_ = evaluate(f"HGB leaves {hp['max_leaf_nodes']} lr {hp['learning_rate']} trees {stg}", raws[k])
                cfg_rec = {"max_leaf_nodes": int(hp["max_leaf_nodes"]), "learning_rate": float(hp["learning_rate"]),
                           "trees": stg, "fit_s": fit_s, "n_iter": int(mdl.n_iter_), "s2a": ev_["info"]}
                sel.append(cfg_rec)
                if best is None or ev_["info"]["bits_per_position"] < best["info"]["bits_per_position"]:
                    best = {**ev_, "cfg": cfg_rec, "ens": ens, "stage": stg}
                del ev_
            del raws, mdl
            gc.collect()
        gate = {"kind": "hgb", "ens": best["ens"], "stage": best["stage"]}
        finish_gate("HGB", sel, best, gate, t_gate, extra={
            "final_fit_capped": False,
            "iteration_cap": "not applicable: the tree counts are configurations (early stopping off), not a convergence cap"})
        del best, sel, XH
    # ---------------- outputs
    if val is not None and per_tok:
        od = out_dir(args)
        np.savez_compressed(od / "validation_bits.npz", doc=val.doc.astype(np.int32), core=core_bits_pos,
                            **{k: v for k, v in per_tok.items()})
        db = {k: np.bincount(val.doc, weights=v, minlength=doc_bytes.shape[0]) for k, v in per_tok.items()}
        controls = [g for g in ("LR", "LR2", "HGB", "C0") if g in per_tok]
        if "TM" in per_tok and controls:
            bpbs = {g: float(per_tok[g].sum() / n_bytes) for g in controls}
            crit = {g: float(results[g]["s2b_heldout"]["bits_per_position"]) for g in controls}
            best_c = min(crit, key=crit.get)          # docs/13 A5 D0.2: chosen on residue 2 (S2b), never on validation
            for g in controls:
                bs = protocol.bootstrap_bpb(db["TM"], doc_bytes, BIDX, doc_bits_ref=db[g])
                M(f"contrast.TM_minus_{g}", bs, label="derived")
            bs = protocol.bootstrap_bpb(db["TM"], doc_bytes, BIDX, doc_bits_ref=db[best_c])
            M("seed_verdict.H9_2", {"best_control": best_c, "best_control_rule": "lowest S2b held-out bits per position",
                                    "s2b_bits_per_position": crit, "tm_minus_best": bs,
                                    "verdict": CS.h92_verdict(bs["point"], bs["ci95"]),
                                    "control_validation_bpb": bpbs}, label="derived",
              note="per-seed reading; the reported verdict is the 3-seed mean of the verdicts run (docs/13 A5 D0.2)")
            tmc = results["TM"]["validation_minus_core"]
            M("seed_verdict.H9_1", CS.h91_verdict(tmc["point"], tmc["ci95"]), label="derived",
              note="per-seed reading; the reported verdict is the verdicts run over seeds 1, 2, 3")
    M("wall_s", time.time() - T0)
    if SMOKE or runlog.current_run_dir() is None:
        # unlogged smoke: keep the metrics next to validation_bits.npz so that the verdicts mode can be exercised
        (out_dir(args) / "metrics_unlogged.json").write_text(json.dumps(_plain(METRICS), indent=1, sort_keys=True))
    return 1 if FAILED else 0


def clause_stats(st: Store, cs: "CS.ClauseSet", fire: np.ndarray, firep: np.ndarray, blocks: int = 1 << 16) -> None:
    """Firing and label-1 firing counts of every clause over all pairs of a store (chunked over blocks)."""
    nb = st.cols.shape[0]
    for b0 in range(0, nb, blocks):
        b1 = min(nb, b0 + blocks)
        n = min(st.n_pairs, b1 * 64) - b0 * 64
        CS.k_clause_stats(np.ascontiguousarray(st.cols[b0:b1]), st.labbits[b0:b1], n, cs.ptr, cs.lits, cs.negs, fire, firep)


def tm_rules(cs: "CS.ClauseSet", s2: Store, val: Store | None, lits, label: str) -> None:
    """The 10 most important clauses of the chosen TM (|signed weight| x firing rate on the S2 pairs),
    as readable rules with firing rate and precision (share of label-1 pairs among the firing pairs)."""
    t0 = time.time()
    fire = np.zeros(cs.n, np.int64)
    firep = np.zeros(cs.n, np.int64)
    clause_stats(s2, cs, fire, firep)
    rate = fire / s2.n_pairs
    imp = np.abs(cs.w) * rate
    top = np.argsort(-imp, kind="stable")[:10]
    lab_s2 = float(int(np.sum(np.asarray(s2.lab[:], dtype=np.int64))) / max(1, s2.n_pairs))
    sub = CS.ClauseSet(ptr=np.concatenate([[0], np.cumsum(np.diff(cs.ptr)[top])]).astype(np.int64),
                       lits=np.concatenate([cs.lits[cs.ptr[j]:cs.ptr[j + 1]] for j in top]).astype(np.int64),
                       negs=np.concatenate([cs.negs[cs.ptr[j]:cs.ptr[j + 1]] for j in top]).astype(np.uint8),
                       w=cs.w[top], origin=cs.origin[top])
    fv = np.zeros(sub.n, np.int64)
    fvp = np.zeros(sub.n, np.int64)
    if val is not None:
        clause_stats(val, sub, fv, fvp)
    rules = []
    for k, j in enumerate(top):
        rules.append({"rank": k + 1, "signed_weight": int(cs.w[j]), "votes": "for copying" if cs.w[j] > 0 else "against copying",
                      "bank": {"class": int(cs.origin[j, 0]), "clause": int(cs.origin[j, 1])},
                      "n_literals": int(cs.ptr[j + 1] - cs.ptr[j]), "importance": float(imp[j]),
                      "s2_firing_rate": float(rate[j]), "s2_precision": float(firep[j] / fire[j]) if fire[j] else None,
                      "validation_firing_rate": float(fv[k] / val.n_pairs) if val is not None else None,
                      "validation_precision": float(fvp[k] / fv[k]) if (val is not None and fv[k]) else None,
                      "rule": CS.rule_text(lits, cs.lits[cs.ptr[j]:cs.ptr[j + 1]], cs.negs[cs.ptr[j]:cs.ptr[j + 1]])})
    M("TM.top10_clauses", rules, label=label,
      note="importance = |signed weight| x firing rate over the S2a pairs; precision = share of pairs with the candidate equal "
           f"to the next token among the pairs where the clause fires; base rate of label 1 over S2a pairs {lab_s2:.6f}")
    M("TM.rules_s", time.time() - t0)
    for r in rules:
        log(f"  #{r['rank']} w {r['signed_weight']:+d} ({r['votes']}) fires {r['s2_firing_rate']:.5f} "
            f"prec {r['s2_precision']} IF {r['rule']}")


# ------------------------------------------------------------------ verdicts
def mode_verdicts(cfg: dict, args) -> int:
    """H9.1, H9.2, gate G1 T2 and the A5 labels (D0.1 band, D0.3 credit) from the three seed runs."""
    vd = cfg["verdicts"]
    runs = [ROOT / r for r in vd["seed_runs"]]
    M("config.params", cfg, label="unverified")
    seeds, rows, gates_rec = [], {}, {}
    doc = None
    for r in runs:
        rj = json.loads((r / "run.json").read_text())
        check(f"{r.name}.exit_0", rj.get("exit_status") == 0, exit_status=rj.get("exit_status"))
        z = np.load(r / "validation_bits.npz")
        s = int(rj["command"][rj["command"].index("--seed") + 1])
        seeds.append(s)
        rows[s] = {k: z[k] for k in z.files}
        mfile = r / "metrics.json"
        if not mfile.exists() and args.allow_unlogged_seed_runs:
            mfile = r / "metrics_unlogged.json"          # smoke of this mode only (unlogged seed outputs)
        mj = json.loads(mfile.read_text())
        gates_rec[s] = {g: mj[f"gate.{g}"]["value"] for g in ("TM", "LR", "LR2", "HGB", "C0")}
        fails = sorted(k for k, v in mj.items() if k.startswith("check.") and v.get("value") is not True)
        check(f"seed{s}.all_checks_true", not fails, failed=fails)
        doc = z["doc"].astype(np.int64) if doc is None else doc
        check(f"seed{s}.same_positions", bool(np.array_equal(z["doc"].astype(np.int64), doc)))
    check("seeds_are_1_2_3", sorted(seeds) == [1, 2, 3], seeds=seeds)
    seeds = sorted(seeds)
    pdir = ROOT / cfg["pairs_run"] / "pairs"
    smeta = json.loads((pdir / "store_meta.json").read_text())
    if runlog.current_run_dir() is not None and not SMOKE:
        mf = hashing.sha256_file(ROOT / cfg["pairs_run"] / "MANIFEST.json")
        check("pairs_manifest_sha256_as_frozen", mf == cfg["pairs_manifest_sha256"], got=mf)
        hashes = protocol.dataset_hashes(("train", "validation"), protocol.CORPUS, V)
        hashes["cvtm_cache_meta_json"] = smeta["cache_meta_sha256"]
        hashes["copy_selector_pairs_manifest"] = mf
        runlog.write_dataset_hashes(hashes, on_conflict="replace")
        M("dataset_hashes", hashes, note="as the seed runs")
    doc_bytes = np.load(pdir / "val" / "doc_bytes.npy").astype(np.float64)
    n_bytes = int(smeta["validation_n_bytes"])
    idx = protocol.bootstrap_index(doc_bytes.shape[0])
    nd = doc_bytes.shape[0]

    def dbits(v):
        return np.bincount(doc, weights=v, minlength=nd)

    core = rows[seeds[0]]["core"]
    for s in seeds:
        check(f"seed{s}.core_bits_identical", bool(np.array_equal(rows[s]["core"], core)))
    core_bpb = float(core.sum() / n_bytes)
    check("core_bpb_equals_L0", core_bpb == float(cfg["core_validation_bpb"]), got=core_bpb)
    gates = ["TM", "LR", "LR2", "HGB", "C0"]
    controls = ["LR", "LR2", "HGB", "C0"]
    table = {}
    for g in gates:
        vals = [float(rows[s][g].sum() / n_bytes) for s in seeds]
        table[g] = {"per_seed": dict(zip(seeds, vals)), "mean": float(np.mean(vals)), "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "s2b_heldout_bits_per_position": {s: gates_rec[s][g]["s2b_heldout"]["bits_per_position"] for s in seeds},
                    "chosen_configuration": {s: {k: v for k, v in gates_rec[s][g]["chosen"].items()
                                                 if k in ("T", "s", "epoch", "l2", "max_leaf_nodes", "learning_rate", "trees",
                                                          "nonempty_clauses", "nit", "converged", "hit_iteration_cap")}
                                             for s in seeds}}
    M("rows.validation_bits_per_byte", {"core": core_bpb, **table},
      note="core + gate read-out; seeds 1, 2, 3 (docs/13 section 5: mean, min and max across seeds)")
    # ---------------- H9.1 and the D0.1 band (G1 T2)
    h91 = {}
    gains, excl = [], []
    for s in seeds:
        bs = protocol.bootstrap_bpb(dbits(rows[s]["TM"]), doc_bytes, idx, doc_bits_ref=dbits(core))
        v = CS.h91_verdict(bs["point"], bs["ci95"])
        h91[s] = {**v, "difference": bs}
        gains.append(-bs["point"])
        excl.append(bool(bs["ci95"][1] < 0.0))
    met = all(h91[s]["met"] for s in seeds)
    mean_tm = np.mean([rows[s]["TM"] for s in seeds], axis=0)
    bs_mean = protocol.bootstrap_bpb(dbits(mean_tm), doc_bytes, idx, doc_bits_ref=dbits(core))
    M("H9_1", {"met": bool(met), "per_seed": h91, "seed_mean_difference": bs_mean,
               "threshold": "gain >= 0.010 bits per byte on every seed, document-bootstrap CI of the difference excluding 0",
               "core_bpb": core_bpb}, label="derived")
    g1 = CS.g1_t2(gains, excl)
    M("G1_T2", g1, label="derived")
    # ---------------- H9.2 (docs/13 A5 D0.2): per seed, the best control is the one with the lowest S2b held-out
    # bits per position (residue 2, never validation); the verdict uses the 3-seed mean of (TM minus that control)
    h92 = {}
    best_of = {}
    for s in seeds:
        crit = {g: float(gates_rec[s][g]["s2b_heldout"]["bits_per_position"]) for g in controls}
        bc = min(crit, key=crit.get)
        best_of[s] = bc
        bs = protocol.bootstrap_bpb(dbits(rows[s]["TM"]), doc_bytes, idx, doc_bits_ref=dbits(rows[s][bc]))
        per_c = {g: protocol.bootstrap_bpb(dbits(rows[s]["TM"]), doc_bytes, idx, doc_bits_ref=dbits(rows[s][g]))
                 for g in controls}
        h92[s] = {"best_control": bc, "s2b_bits_per_position": crit, "tm_minus_best": bs,
                  "verdict_this_seed": CS.h92_verdict(bs["point"], bs["ci95"]), "tm_minus_each_control": per_c}
    diff_tm = np.mean([rows[s]["TM"] for s in seeds], axis=0)
    diff_bc = np.mean([rows[s][best_of[s]] for s in seeds], axis=0)
    bsm = protocol.bootstrap_bpb(dbits(diff_tm), doc_bytes, idx, doc_bits_ref=dbits(diff_bc))
    verdict = CS.h92_verdict(bsm["point"], bsm["ci95"])
    capped = {s: {g: bool(gates_rec[s][g].get("final_fit_capped", False)) for g in controls} for s in seeds}
    any_capped = any(v for d in capped.values() for v in d.values())
    verdict_reported = verdict
    if verdict == "beats" and any_capped:
        verdict_reported = "matches (beats not claimed: a control fit still stopped at its raised iteration cap, D0.2)"
    M("H9_2", {"verdict": verdict_reported, "verdict_rule_output": verdict, "tm_minus_best_3seed_mean": bsm,
               "best_control_per_seed": best_of, "per_seed": h92, "controls_final_fit_capped": capped,
               "rule": "docs/13 A4 H9.2 with A5 D0.2: per seed the best control is chosen on residue 2 (lowest S2b held-out "
                       "bits per position); the verdict uses the 3-seed mean of (TM minus best control) with a paired "
                       "document bootstrap: at most -0.002 with the CI excluding 0 = beats; at most 0.000 = matches; "
                       "else trails"}, label="derived")
    # ---------------- TM minus C0 (its own row, D0.2) and the D0.3 credit
    c0m = np.mean([rows[s]["C0"] for s in seeds], axis=0)
    bs_c0 = protocol.bootstrap_bpb(dbits(diff_tm), doc_bytes, idx, doc_bits_ref=dbits(c0m))
    per_c0 = {s: protocol.bootstrap_bpb(dbits(rows[s]["TM"]), doc_bytes, idx, doc_bits_ref=dbits(rows[s]["C0"]))
              for s in seeds}
    M("TM_minus_C0", {"three_seed_mean": bs_c0, "per_seed": per_c0}, label="derived",
      note="docs/13 A5 D0.2: reported as its own row")
    M("D0_3_credit", CS.credit_d03(g1["band"], bs_c0["point"], bs_c0["ci95"]), label="derived")
    # every control against the TM, 3-seed mean
    each = {}
    for g in controls:
        gm = np.mean([rows[s][g] for s in seeds], axis=0)
        each[g] = protocol.bootstrap_bpb(dbits(diff_tm), doc_bytes, idx, doc_bits_ref=dbits(gm))
    M("TM_minus_each_control_3seed_mean", each, label="derived")
    for k in ("rows.validation_bits_per_byte", "H9_1", "G1_T2", "H9_2", "TM_minus_C0", "D0_3_credit"):
        log(f"{k}: {json.dumps(_plain(METRICS[k]['value']), sort_keys=True)}")
    M("wall_s", time.time() - T0)
    return 1 if FAILED else 0


# ------------------------------------------------------------------ main
def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "seed", "verdicts"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke-docs", type=int, default=0, help="smoke: first k documents per group (unlogged numbers)")
    ap.add_argument("--pairs-dir", default=None, help="smoke: a pairs directory outside results/")
    ap.add_argument("--gates", default=None, help="smoke: comma-separated subset of the gates")
    ap.add_argument("--no-validation", action="store_true", help="pilot: never score validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-unlogged-seed-runs", action="store_true",
                    help="smoke of the verdicts mode: read metrics_unlogged.json of unlogged seed outputs")
    args = ap.parse_args()
    SMOKE = bool(args.smoke_docs or args.pairs_dir or args.gates or args.allow_unlogged_seed_runs)
    cfg = load_cfg(args.config)
    if args.mode == "build":
        rc = mode_build(cfg, args)
    elif args.mode == "seed":
        if args.seed is None:
            raise SystemExit("--seed is required")
        rc = mode_seed(cfg, args)
    else:
        rc = mode_verdicts(cfg, args)
    if FAILED:
        log(f"FAILED checks: {FAILED}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
