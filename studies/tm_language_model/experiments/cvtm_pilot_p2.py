"""Pilot P2 of docs/13 amendment A5 (EXPLORATORY; counting only, no TM; supports no reported claim):
the ceiling of class and skip templates at full data.

Registered text (docs/13 A5, P2): templates c2_1, c2_21, t1_c2_2, t1_c3_2, c3_321, t21_c2_3, t32_skip and
t321_c2_4, counted over the training split without H'; class maps: the repository's exchange clusters
(243 and 2,497), PPMI-SVD k-means (256 and 2,048), and a random partition of equal size; mixed into the
frozen core with k*-by-count-bucket weights fitted on residue 2a, judged on residue 2b. G_best is the
best real map's gain; R is G_best minus the random map's gain. NO-GO: G_best below 0.005 or R below
0.003. GO: G_best at least 0.010 with the CI excluding 0. In between otherwise. Prediction NO-GO 0.8.

Definitions made here (every one in src/tmlm/cvtm/templates.py, docstring):
- templates, packing and the estimator; the two templates that are not in the source script;
- the PPMI-SVD map and the random partitions (one per real map family, the real labels permuted, seed 100);
- residue 2a / 2b: residue-2 documents by ascending stable ID, even ranks 2a, odd ranks 2b;
- R uses the random partition of the SAME sizes as the best real map ("equal size"); R against the
  better of the two random partitions is logged too (conservative reading);
- the base is the frozen CVTM-core over the KN-5 fitted without H' (cache array core_p_gold of the H'
  group) with k* = kstar_noHp; bits per byte use the UTF-8 bytes of the 2b documents from the raw
  train store;
- bootstrap: residue-2b documents, 2,000 resamples, numpy SeedSequence([100, 20260924, 2]), per
  resample the ratio of summed bits to summed bytes (`tmlm.cvtm.protocol.bootstrap_bpb`).

Inputs: the train store at V 8000 (tokens, document starts, kind_map), the raw train store's per-
document bytes, and the H' group of the frozen cache `results/20260924T085728Z_cvtm_cache_build/`.
Validation and test are never loaded (this script never calls load_eval_split or reads the cache's
validation group).

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1
    scripts/cpu_slot.sh 1 -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py \\
        --no-gpu-sampler --name cvtm_pilot_p2_exploratory --config experiments/cvtm/configs/cvtm_pilot_p2.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_pilot_p2.py --config experiments/cvtm/configs/cvtm_pilot_p2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing  # noqa: E402
from tmlm.cvtm import cache as C  # noqa: E402
from tmlm.cvtm import protocol as P  # noqa: E402
from tmlm.cvtm import templates as T  # noqa: E402

T0 = time.time()
C0 = time.process_time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[p2 {time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - T0:7.0f}s cpu {time.process_time() - C0:7.0f}s] {msg}",
          flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}


def flush() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


def check(name: str, ok: bool, **details) -> None:
    M(f"check.{name}", bool(ok), details=details)
    log(f"check {name} = {bool(ok)} {details if details else ''}")
    if not ok:
        flush()
        raise SystemExit(f"pipeline check failed: {name} {details}")


def rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="counting and PPMI over the first --smoke-docs kept train documents")
    ap.add_argument("--smoke-docs", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=None, help="smoke only: override params.lbfgs_iters")
    ap.add_argument("--out", default=None, help="output dir for per-position arrays (default: the run dir)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    prm = cfg["params"]
    seed = int(prm["seed"])
    M("args", vars(args))
    M("mode_smoke", bool(args.smoke))
    out_dir = Path(args.out or os.environ.get("TMLM_RUN_DIR") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    from tmlm.data.corpus_store import load_split, raw_dir

    tr = load_split(P.CORPUS, P.VOCAB, "train", check=not args.smoke)
    tok = np.asarray(tr.tok)
    ds = np.asarray(tr.doc_start).astype(np.int64)
    km = np.asarray(tr.kind_map)
    n_docs = int(ds.shape[0] - 1)
    res_tr = P.residues(n_docs)
    in_hp = np.isin(res_tr, np.asarray(P.HPRIME_RESIDUES))
    keep = ~in_hp
    if args.smoke:
        kept_idx = np.flatnonzero(keep)[: args.smoke_docs]
        keep = np.zeros(n_docs, bool)
        keep[kept_idx] = True
    M("data.train_tokens", int(ds[-1]))
    M("data.train_docs", n_docs)
    M("data.counting_docs", int(keep.sum()), note="training documents without H' (residues 0, 1, 2)")
    M("data.counting_tokens", int(np.diff(ds)[keep].sum()))
    if not args.smoke:
        h = P.write_dataset_hashes(splits=("train",))
        M("data.dataset_hashes", h)
    log(f"train: {n_docs} docs, {int(ds[-1])} tokens; counting docs {int(keep.sum())}, tokens {int(np.diff(ds)[keep].sum())}")

    cache_dir = ROOT / prm["cache_run"] / "cache"
    meta = C.read_meta(cache_dir)
    names = ["gold", "doc_start", "doc_index", "residue", "core_p_gold", "kstar_noHp"]
    A = C.load_arrays(cache_dir, "hprime", names)
    for nm in names:
        rec = meta["files"].get(f"hprime/{nm}.npy")
        if rec is None:
            check(f"cache.hprime.{nm}.listed", False)
        if not args.smoke:
            check(f"cache.hprime.{nm}.sha256", hashing.sha256_file(cache_dir / "hprime" / f"{nm}.npy") == rec["sha256"])
    M("cache.meta_sha256", hashing.sha256_file(cache_dir / C.META_FILE))
    gold = np.asarray(A["gold"]).astype(np.int64)
    hds = np.asarray(A["doc_start"]).astype(np.int64)
    hdi = np.asarray(A["doc_index"]).astype(np.int64)
    hres = np.asarray(A["residue"]).astype(np.int64)
    pcore = np.asarray(A["core_p_gold"]).astype(np.float64)
    kstar = np.asarray(A["kstar_noHp"]).astype(np.int64)
    check("hprime.doc_index_is_Hprime", np.array_equal(hdi, np.flatnonzero(in_hp)),
          n_docs=int(hdi.shape[0]), n_expected=int(in_hp.sum()))
    check("hprime.residue_equals_protocol", np.array_equal(hres, P.residues(hdi)))
    lens = np.diff(ds)[hdi]
    check("hprime.doc_lengths_equal_train", np.array_equal(np.diff(hds), lens))
    tr_hp = np.concatenate([tok[ds[i]:ds[i + 1]] for i in hdi]).astype(np.int64)
    check("hprime.gold_equals_train_tokens", np.array_equal(tr_hp, gold), n=int(gold.shape[0]))
    del tr_hp
    check("hprime.core_p_gold_in_0_1", bool(np.all((pcore > 0) & (pcore <= 1))))
    check("hprime.kstar_in_0_4", bool(kstar.min() >= 0 and kstar.max() <= 4))

    a_doc, b_doc = T.split_2ab(hdi, hres)
    pos_doc = np.repeat(np.arange(hdi.shape[0]), np.diff(hds))
    rows = np.flatnonzero(hres[pos_doc] == 2)                  # residue-2 positions (the evaluation rows)
    is_a = a_doc[pos_doc[rows]]
    is_b = b_doc[pos_doc[rows]]
    check("split.2a_2b_partition_residue2", bool(np.all(is_a ^ is_b)))
    doc_bytes_all = np.diff(np.load(raw_dir(P.CORPUS, "train") / "doc_bytes.npy").astype(np.int64))
    check("bytes.train_doc_bytes_per_token_doc", doc_bytes_all.shape[0] == n_docs)
    b_docs = np.flatnonzero(b_doc)                               # local H' doc numbers of 2b
    a_docs = np.flatnonzero(a_doc)
    bytes_b = doc_bytes_all[hdi[b_docs]].astype(np.float64)
    bytes_a = doc_bytes_all[hdi[a_docs]].astype(np.float64)
    M("split.2a", {"n_docs": int(a_docs.shape[0]), "n_positions": int(is_a.sum()), "n_bytes": int(bytes_a.sum()),
                   "doc_index_sha256": P.index_sha256(hdi[a_docs])})
    M("split.2b", {"n_docs": int(b_docs.shape[0]), "n_positions": int(is_b.sum()), "n_bytes": int(bytes_b.sum()),
                   "doc_index_sha256": P.index_sha256(hdi[b_docs])})
    log(f"residue 2: {rows.shape[0]} positions; 2a {int(a_docs.shape[0])} docs {int(is_a.sum())} pos; "
        f"2b {int(b_docs.shape[0])} docs {int(is_b.sum())} pos")

    # 2b document of every evaluation row (for the bootstrap)
    b_local = np.full(hdi.shape[0], -1, np.int64)
    b_local[b_docs] = np.arange(b_docs.shape[0])
    row_bdoc = b_local[pos_doc[rows]]
    BIDX = P.bootstrap_index(b_docs.shape[0], n_boot=int(prm["bootstrap_resamples"]), seed=tuple(prm["bootstrap_seed"]))

    pb = pcore[rows]
    core_bits = -np.log2(pb)
    core_bpb_b = float(core_bits[is_b].sum() / bytes_b.sum())
    core_bpb_a = float(core_bits[is_a].sum() / bytes_a.sum())
    M("core.2b.bits_per_byte", core_bpb_b, note="frozen CVTM-core over the KN-5 without H' (cache core_p_gold, hprime)")
    M("core.2a.bits_per_byte", core_bpb_a)
    log(f"core bpb: 2a {core_bpb_a:.5f}, 2b {core_bpb_b:.5f}")
    flush()

    def doc_bits_b(bits_rows: np.ndarray) -> np.ndarray:
        return np.bincount(row_bdoc[is_b], weights=bits_rows[is_b], minlength=b_docs.shape[0])

    core_doc_b = doc_bits_b(core_bits)

    # ------------------------------------------------------------------ class maps
    t1 = time.time()
    maps: dict[str, T.ClassMap] = {"exchange": T.exchange_map(km)}
    cm_ppmi, pinfo = T.ppmi_svd_map(tok, ds, keep, P.VOCAB, sizes=tuple(prm["ppmi_sizes"]), dims=int(prm["ppmi_dims"]),
                                     seed=seed)
    maps["ppmi"] = cm_ppmi
    M("map.ppmi.fit", {**pinfo, "wall_s": time.time() - t1}, label="measured")
    log(f"PPMI-SVD map: {pinfo['kmeans_coarse']} / {pinfo['kmeans_fine']} ({time.time() - t1:.0f} s)")
    maps["random_exchange"] = T.random_like(maps["exchange"], (seed, 1))
    maps["random_ppmi"] = T.random_like(maps["ppmi"], (seed, 2))
    for k, cm in maps.items():
        M(f"map.{k}", {"k_coarse": cm.k_coarse, "k_fine": cm.k_fine,
                       "coarse_sha256": hashing.sha256_array(cm.coarse.astype(np.int64)),
                       "fine_sha256": hashing.sha256_array(cm.fine.astype(np.int64)),
                       "largest_coarse_class": int(np.bincount(cm.coarse).max()),
                       "largest_fine_class": int(np.bincount(cm.fine).max())})
    if not args.smoke:
        check("map.exchange_sizes_243_2497", (maps["exchange"].k_coarse, maps["exchange"].k_fine) == (243, 2497))
    np.savez_compressed(out_dir / "class_maps.npz", **{f"{k}_{r}": getattr(cm, r) for k, cm in maps.items() for r in ("coarse", "fine")})
    flush()

    # ------------------------------------------------------------------ template counts
    counts: dict[str, dict[str, T.TemplateCounts]] = {k: {} for k in maps}
    shared: dict[str, T.TemplateCounts] = {}
    for name in T.TEMPLATE_NAMES:
        fams = list(maps) if T.uses_map(name) else [None]
        for fam in fams:
            t1 = time.time()
            tc = T.count_template(name, maps[fam] if fam else None, P.VOCAB, gold, hds, rows, tok, ds, keep)
            key = fam or "shared"
            if fam is None:
                shared[name] = tc
                for k in maps:
                    counts[k][name] = tc
            else:
                counts[fam][name] = tc
            fire_b = float(tc.fires[is_b].mean())
            M(f"template.{key}.{name}", {"fire_rate_2b": fire_b, "fire_rate_2a": float(tc.fires[is_a].mean()),
                                         "median_N_when_firing_2b": float(np.median(tc.n[is_b & tc.fires])) if (is_b & tc.fires).any() else 0.0,
                                         "gold_seen_rate_2b": float((tc.c[is_b] > 0).mean()),
                                         "eval_contexts": tc.n_contexts_eval, "counting_positions": tc.counting_positions,
                                         "wall_s": time.time() - t1})
            if not args.smoke:
                check(f"template.{key}.{name}.counted_all_kept_positions",
                      tc.counting_positions == int(np.diff(ds)[keep].sum()))
            log(f"template {name:<10} {key:<16} fires 2b {fire_b:.3f} ({time.time() - t1:.1f} s)")
        flush()

    # ------------------------------------------------------------------ mixtures
    iters = int(args.iters if args.iters is not None else prm["lbfgs_iters"])
    if args.iters is not None and not args.smoke:
        raise SystemExit("--iters is a smoke-only override")
    M("mix.lbfgs_iters_cap", iters)
    per_pos: dict[str, np.ndarray] = {"core": core_bits[is_b]}

    def mix(tag: str, tcs: list[T.TemplateCounts]) -> dict:
        t1 = time.time()
        r, b, p, nb = T.components(tcs, kstar[rows])
        th, res = T.fit_mix(pb, r, b, p, nb, is_a, iters)
        bits = T.eval_mix(pb, r, b, p, th)
        bpb_b = float(bits[is_b].sum() / bytes_b.sum())
        bpb_a = float(bits[is_a].sum() / bytes_a.sum())
        db = doc_bits_b(bits)
        gain = P.bootstrap_bpb(core_doc_b, bytes_b, BIDX, doc_bits_ref=db)    # core minus mix: positive = gain
        out = {"bpb_2b": bpb_b, "bpb_2a_in_sample": bpb_a, "gain_2b": gain["point"], "gain_2b_ci95": gain["ci95"],
               "gain_2a_in_sample": core_bpb_a - bpb_a, "n_buckets": int(nb), "buckets_fitted": int((np.bincount(b[is_a[r]], minlength=nb) > 0).sum()),
               "lbfgs_nit": int(res.nit), "lbfgs_converged": bool(res.success), "lbfgs_message": str(res.message)[:80],
               "wall_s": time.time() - t1}
        if tag.startswith("all8."):
            per_pos[tag] = bits[is_b]
        log(f"mix {tag:<32} 2b bpb {bpb_b:.5f} gain {gain['point']:+.5f} [{gain['ci95'][0]:+.5f}, {gain['ci95'][1]:+.5f}] "
            f"nit {res.nit}{'' if res.success else ' NOT CONVERGED'} ({time.time() - t1:.0f} s)")
        out["_doc_bits"] = db
        return out

    fam_rows = {}
    for fam in maps:
        o = mix(f"all8.{fam}", [counts[fam][n] for n in T.TEMPLATE_NAMES])
        fam_rows[fam] = o
        M(f"mix.all8.{fam}", {k: v for k, v in o.items() if not k.startswith("_")}, label="measured",
          note="gain = core minus mixture on residue 2b, bits per byte; CI: document bootstrap (derived)")
        flush()

    real = ("exchange", "ppmi")
    best = max(real, key=lambda f: fam_rows[f]["gain_2b"])
    best_2a = max(real, key=lambda f: fam_rows[f]["gain_2a_in_sample"])
    rnd = f"random_{best}"
    g_best = fam_rows[best]["gain_2b"]
    g_best_ci = fam_rows[best]["gain_2b_ci95"]
    rdiff = P.bootstrap_bpb(fam_rows[rnd]["_doc_bits"], bytes_b, BIDX, doc_bits_ref=fam_rows[best]["_doc_bits"])
    r_val = rdiff["point"]
    rnd_max = max(("random_exchange", "random_ppmi"), key=lambda f: fam_rows[f]["gain_2b"])
    rdiff_c = P.bootstrap_bpb(fam_rows[rnd_max]["_doc_bits"], bytes_b, BIDX, doc_bits_ref=fam_rows[best]["_doc_bits"])
    v = T.verdict(g_best, g_best_ci, r_val)
    v_c = T.verdict(g_best, g_best_ci, rdiff_c["point"])
    M("P2.G_best", g_best, label="derived", family=best, ci95=g_best_ci)
    M("P2.best_family_by_2a", best_2a, label="derived", note="the real map with the larger in-sample 2a gain (selection-free reading)")
    M("P2.R", r_val, label="derived", ci95=rdiff["ci95"], random_family=rnd,
      note="G_best minus the gain of the random partition of the same sizes as the best real map")
    M("P2.R_vs_best_random", rdiff_c["point"], label="derived", ci95=rdiff_c["ci95"], random_family=rnd_max,
      note="conservative reading: against the better random partition")
    M("P2.verdict", v["verdict"], label="derived", details=v,
      rule="A5 P2: NO-GO if G_best < 0.005 or R < 0.003; GO if G_best >= 0.010 with the CI excluding 0; else in between")
    M("P2.verdict_conservative_R", v_c["verdict"], label="derived", details=v_c)
    log(f"G_best {g_best:+.5f} ({best}) CI {g_best_ci}; R {r_val:+.5f} CI {rdiff['ci95']} vs {rnd}; "
        f"R_conservative {rdiff_c['point']:+.5f} vs {rnd_max}; verdict {v['verdict']} (conservative {v_c['verdict']})")
    flush()

    # ------------------------------------------------------------------ diagnostics: each template alone
    if prm.get("single_template_diagnostics", True):
        for fam in ("exchange", "ppmi", rnd):
            for name in T.TEMPLATE_NAMES:
                if fam != "exchange" and not T.uses_map(name):
                    continue      # t32_skip is map-free: one row suffices
                o = mix(f"single.{fam}.{name}", [counts[fam][name]])
                M(f"mix.single.{fam}.{name}", {k: v for k, v in o.items() if not k.startswith("_")}, label="measured",
                  note="diagnostic: this template alone mixed into the core")
            flush()
    if prm.get("leave_one_out_diagnostics", False):
        # leave-one-out on the best real map
        for name in T.TEMPLATE_NAMES:
            o = mix(f"loo.{best}.{name}", [counts[best][n] for n in T.TEMPLATE_NAMES if n != name])
            M(f"mix.leave_one_out.{best}.{name}",
              {**{k: v for k, v in o.items() if not k.startswith("_")},
               "loss_vs_all8_2b": o["bpb_2b"] - fam_rows[best]["bpb_2b"]}, label="measured",
              note="diagnostic: all templates but this one")
        flush()

    np.savez_compressed(out_dir / "per_position_bits_2b.npz", row_bdoc=row_bdoc[is_b],
                        **{k.replace(".", "__"): v for k, v in per_pos.items()})
    M("peak_rss_gib", rss_gib())
    M("wall_s", time.time() - T0)
    M("cpu_s", time.process_time() - C0)
    flush()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
