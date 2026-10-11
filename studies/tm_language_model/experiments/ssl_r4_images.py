"""R4 image domain (docs/13 amendment A7, hypothesis H17.10): does the same docs/11 engine, trained self-supervised
by masked patch prediction on Fashion-MNIST patch tokens, learn clauses whose firing pattern is a useful image
representation for few-label classification, beyond what counting gives?

Data (src/tmlm/ssl/patch_tokens.py): the Fashion-MNIST TRAIN split only (60,000 images, pinned Hugging Face
revision, MIT); a hash-carved dev set of exactly 10,000 images; the test split is never downloaded. Binarised at
> 75, 4 x 4 patches coded to V = 2,000 (1,999 codes + ROW_END), 56 tokens per image, the 64 most frequent tokens are
stop tokens, kind_map from the repository's exchange clustering on the pool bigrams (as for text).

Stages (each a logged run; configs experiments/cvtm/configs/ssl_r4_images*.json):
  data      download the pinned train parquet file (only that file), check its sha256 against the hub's LFS oid,
            build the corpus store (create-only, data/<corpus>/bpe2000/train/); with --write-prereg-hash write
            docs/notes/preregistration-data-hashes-A7-fashion_mnist.json once (an existing file is verified, never
            rewritten)
  gpu       (CUDA engine, shared lock) per arm: masked patch prediction with lever F1 (A7: keep target p w.p.
            min(1, sqrt(t/f(y)) + t/f(y)), t = 1e-4, draw by mix64(p)), C = 65,536 clauses, engine defaults of
            docs/11 section 7 except n_bag_k = 3 (a document has 56 tokens: buckets 3 and up cover distances above
            64 and can never be true inside an image), E epochs; arm "main" uses the store's kind_map and saves the
            YOUNG bank (C0y: the same run stopped after 1 percent of the batches of epoch 1); arm "randkm" uses a
            random kind_map (token rows permuted, patch_tokens.random_kind_map; concepts removed). After training:
            the image code of every image (pool and dev) for the banks main.final, main.young, randkm.final = the
            counts of (fired clause, image quadrant) over the 49 patch positions, forward pass with only the target
            position hidden; masked patch top-1 on the dev patches under a frozen 15 percent mask (diagnostic).
  probe     (CPU) few-label probes: 10 and 100 labels per class, 5 stratified draws per budget, labels drawn from
            the pool only, accuracy on the 10,000 dev images. Primary probe: L2 logistic regression, C by 4-fold CV
            inside the labelled sample (tmlm.ssl.transfer.lr_cell), on log(1 + count) rows scaled to unit L2 norm.
            Representations: code_main, code_young, code_randkm, bovw (bag of visual words per quadrant: counts of
            (code, quadrant)), pixels (grey values), pixels_bin (the binarised pixels the tokens see). Control
            "tm_supervised": a supervised tmu Coalesced TM on the binarised pixels of the same labelled sample
            (configuration by 3-fold CV inside the sample).
  probe_tm  (CPU) the A7 secondary probe: a tmu Coalesced TM on the binarised top-K document-frequency columns
            (over the pool) of code_main, code_young and bovw, one fixed configuration.
  report    merge probe cells over seeds; H17.10: at 10 labels per class, code_main minus the stronger (by mean) of
            code_young and bovw, paired over (seed, draw), at least 2.0 points with the paired-bootstrap 95 percent
            interval (10,000 resamples) excluding 0; everything else reported next to it.

Launch commands (detached, CPU slot FIRST, docs/12 F25; from the study folder after `source env/env.sh`):
  data (once; CPU, about 2 minutes of clustering):
    nohup scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 8 -- \\
        scripts/run_logged.py --no-gpu-sampler --name ssl_r4_data --config experiments/cvtm/configs/ssl_r4_images.json \\
        -- .venv-tm/bin/python -u experiments/ssl_r4_images.py --stage data --config experiments/cvtm/configs/ssl_r4_images.json \\
        --write-prereg-hash > /dev/null 2>&1 &
  gpu, one run per seed (exploratory seed 100 first with ssl_r4_images_seed100.json, then seeds 1, 2, 3):
    nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/mem_guard.py \\
        --scope group --limit-gib 12 -- scripts/run_logged.py --name ssl_r4_gpu_seed1 \\
        --config experiments/cvtm/configs/ssl_r4_images.json -- .venv-tm/bin/python -u experiments/ssl_r4_images.py \\
        --stage gpu --config experiments/cvtm/configs/ssl_r4_images.json --seed 1 > /dev/null 2>&1 &
  probe and probe_tm, one run per gpu run:
    nohup scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 16 -- \\
        scripts/run_logged.py --no-gpu-sampler --name ssl_r4_probe_seed1 --config experiments/cvtm/configs/ssl_r4_images.json \\
        -- .venv-tm/bin/python -u experiments/ssl_r4_images.py --stage probe --config experiments/cvtm/configs/ssl_r4_images.json \\
        --gpu-run <results dir of ssl_r4_gpu_seed1> > /dev/null 2>&1 &
    (the same with --stage probe_tm and --name ssl_r4_probe_tm_seed1)
  report:
    scripts/cpu_slot.sh 1 -- scripts/run_logged.py --no-gpu-sampler --name ssl_r4_report \\
        --config experiments/cvtm/configs/ssl_r4_images.json -- .venv-tm/bin/python -u experiments/ssl_r4_images.py \\
        --stage report --config experiments/cvtm/configs/ssl_r4_images.json --runs <probe dirs and gpu dirs, comma separated>
Smoke: experiments/cvtm/configs/ssl_r4_images_smoke.json (2,000 images, C = 4,096, 1 epoch) with --data-root and
--hf-cache in a scratch directory, so nothing is written under data/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import patch_tokens as PT  # noqa: E402

T0 = time.time()
METRICS: dict = {}
FAILED: list = []
HF_CACHE = ROOT / "data" / "hf_cache" / "hub"
PREREG_FILE = ROOT / "docs" / "notes" / "preregistration-data-hashes-A7-fashion_mnist.json"
EVAL_EPOCH = 0


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}


def flush() -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_metrics(METRICS, on_conflict="replace")


def check(name: str, ok: bool, **extra) -> None:
    M(f"check.{name}", bool(ok), **extra)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {extra}")


def run_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_r4_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def write_cell(cell: dict) -> None:
    with open(run_dir() / "cells.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(cell, sort_keys=True, default=float) + "\n")


def data_root(args) -> Path | None:
    return Path(args.data_root) if args.data_root else None


# ------------------------------------------------------------------ data
def hub_file(cache: Path, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(PT.HF["repo"], PT.HF["file"], repo_type="dataset", revision=PT.HF["revision"],
                                cache_dir=str(cache), local_files_only=offline))


def hub_lfs_sha() -> str | None:
    from huggingface_hub import HfApi

    for e in HfApi().list_repo_tree(PT.HF["repo"], repo_type="dataset", revision=PT.HF["revision"], recursive=True, expand=True):
        if str(e.path) == PT.HF["file"]:
            lfs = getattr(e, "lfs", None)
            return (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs is not None else None
    return None


def stage_data(cfg: dict, args) -> int:
    P = cfg["params"]
    cache = Path(args.hf_cache) if args.hf_cache else HF_CACHE
    path = hub_file(cache, offline=False)
    sha = hashing.sha256_file(path)
    live = hub_lfs_sha()
    check("parquet_sha256_equals_pinned_lfs_oid", sha == PT.HF["lfs_sha256"], sha256=sha, pinned=PT.HF["lfs_sha256"])
    check("parquet_sha256_equals_hub_lfs_oid_now", sha == live, hub=live)
    limit = P.get("limit_images")
    t0 = time.time()
    images, labels = PT.read_fashion_parquet(path, limit=limit)
    M("decode_s", time.time() - t0)
    n = images.shape[0]
    if limit is None:
        check("n_images_60000", n == PT.N_IMAGES, n=n)
    dev = PT.dev_mask(PT.N_IMAGES, PT.DEV_SIZE)[:n]
    t0 = time.time()
    arrays, meta = PT.build_store_arrays(images, labels, dev, n_codes=int(P["n_codes"]), cluster_seed=int(P["cluster_seed"]),
                                         cluster_max_iter=int(P["cluster_max_iter"]), n_stop=int(P["n_stop"]))
    M("build_s", time.time() - t0)
    meta["source"] = {**PT.HF, "file_sha256": sha, "limit_images": limit,
                      "access": "huggingface_hub.hf_hub_download of the train parquet file only; the test split is never downloaded"}
    meta["dev_rule"] = (f"the {PT.DEV_SIZE} image row indices i of the {PT.N_IMAGES}-row train file with the smallest "
                        f"mix64(i XOR {PT.DEV_SALT:#x}) (tmlm.ssl.patch_tokens.dev_mask)")
    meta["tokenization"] = {"thresh": PT.THRESH, "patch": PT.PATCH, "grid": PT.GRID, "tokens_per_image": PT.TOKENS_PER_IMAGE,
                            "row_end": int(P["n_codes"]), "n_codes": int(P["n_codes"]), "n_stop": int(P["n_stop"])}
    stored = PT.write_store(arrays, meta, corpus=P["corpus"], root=data_root(args))
    for k in ("n_images", "n_pool", "n_dev", "distinct_patterns_pool", "pool_patch_coverage", "row_end_is_stop", "n_targets", "bigram_types"):
        M(f"store.{k}", meta[k])
    for lv in (1, 2, 3):
        L = meta["clusters"][f"level{lv}"]
        M(f"store.clusters.level{lv}", {k: L[k] for k in ("empty", "singletons", "largest", "converged")} | {"iterations": len(L["objective"])})
    M("store.class_counts_pool", np.bincount(labels[~dev], minlength=10).tolist())
    M("store.class_counts_dev", np.bincount(labels[dev], minlength=10).tolist())
    M("store.sha256", stored["sha256"])
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({"fashion_mnist_train_parquet": sha, **{f"{P['corpus']}/{k}": v for k, v in stored["sha256"].items()}},
                                    on_conflict="replace")
    if args.write_prereg_hash:
        if limit is not None:
            raise SystemExit("--write-prereg-hash needs the full train file (no limit_images)")
        record = {"purpose": "docs/13 A7 phase R4: Fashion-MNIST train split hashed before any use; the test split is never downloaded",
                  "written_by": "experiments/ssl_r4_images.py --stage data --write-prereg-hash; write-once, never edited",
                  "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "source": meta["source"], "dev_rule": meta["dev_rule"], "tokenization": meta["tokenization"],
                  "store": {"corpus": P["corpus"], "sha256": stored["sha256"]},
                  "dev_rows_sha256": hashing.sha256_bytes(np.flatnonzero(dev).astype("<i8").tobytes()),
                  "row_index_format": "int64 little endian, ascending"}
        if PREREG_FILE.exists():
            old = json.loads(PREREG_FILE.read_text())
            same = (old["source"]["file_sha256"] == sha and old["store"]["sha256"] == stored["sha256"]
                    and old["dev_rows_sha256"] == record["dev_rows_sha256"])
            check("prereg_hash_file_matches", same)
            M("prereg_hash_file", "existing file read and verified; not rewritten")
        else:
            with open(PREREG_FILE, "x", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1, sort_keys=True)
                fh.write("\n")
            M("prereg_hash_file", f"written once: {PREREG_FILE.relative_to(ROOT)}", sha256=hashing.sha256_file(PREREG_FILE))
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ gpu: pretraining and image codes
def label_draws(labels: np.ndarray, pool_img: np.ndarray, seed: int, budgets, draws: int) -> dict:
    """(budget, draw) -> labelled pool image ids; the same function serves the gpu stage (rows to keep) and the probes."""
    return {(int(b), d): PT.stratified_sample(labels, pool_img, int(b), key=(int(seed), int(b), d))
            for b in budgets for d in range(int(draws))}


def image_codes(eng, n_images: int, chunk_images: int, keep: np.ndarray | None = None,
                df_rows: np.ndarray | None = None) -> tuple[F.SparseDocs, dict, np.ndarray]:
    """Counts of (fired clause, quadrant) per image over its 49 patch positions, only the target position hidden.
    Every image is evaluated; the returned SparseDocs holds the rows of the images with keep[i] (default all), in
    ascending image order, counts u16; df = number of images with df_rows[i] (default all) in which each column is
    non-zero (int64 [4C], accumulated on the GPU over every such image, kept or not)."""
    cp = eng.cp
    C = int(eng.cfg.n_clauses)
    keep = np.ones(n_images, bool) if keep is None else np.asarray(keep, bool)
    df_rows = np.ones(n_images, bool) if df_rows is None else np.asarray(df_rows, bool)
    keep_dev, dfr_dev = cp.asarray(keep), cp.asarray(df_rows)
    df = cp.zeros(4 * C, dtype=cp.int64)
    zero_bitmap = cp.zeros(int(eng.corpus.tok.shape[0]), dtype=cp.uint8)
    parts, fired_total, n_pos = [], 0, 0
    t0 = time.time()
    for i0 in range(0, n_images, chunk_images):
        i1 = min(n_images, i0 + chunk_images)
        pos = PT.patch_positions(np.arange(i0, i1))
        pos_dev = cp.asarray(pos)
        eng._sync_state_struct()
        fw = eng._forward(EVAL_EPOCH, pos_dev, zero_bitmap, 0)
        eng._marks = []
        fe, fj = fw.fired_ex.astype(cp.int64), fw.fired_j.astype(cp.uint64)
        n_f = int(fe.size)
        fired_total += n_f
        n_pos += int(pos.shape[0])
        kept = np.flatnonzero(keep[i0:i1])
        if n_f:
            p = pos_dev[fe].astype(cp.int64)
            img = p // PT.TOKENS_PER_IMAGE
            o = p % PT.TOKENS_PER_IMAGE
            q = (2 * ((o // PT.ROW_LEN) >= 4) + ((o % PT.ROW_LEN) >= 4)).astype(cp.uint64)
            key = (img.astype(cp.uint64) << cp.uint64(32)) | (fj * cp.uint64(4) + q)
            key.sort()
            brk = cp.flatnonzero(key[1:] != key[:-1]) + 1
            starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
            uk = key[starts]
            cnt = cp.diff(cp.concatenate([starts, cp.asarray([n_f], dtype=starts.dtype)]))
            uimg = (uk >> cp.uint64(32)).astype(cp.int64)
            ucol = (uk & cp.uint64(0xFFFFFFFF)).astype(cp.int64)
            dsel = dfr_dev[uimg]
            if bool(dsel.any()):
                df += cp.bincount(ucol[dsel], minlength=4 * C)
            ksel = keep_dev[uimg]
            rows_img = uimg[ksel].get()
            ids = ucol[ksel].get().astype(np.uint32)
            c = cnt[ksel].get()
            if c.size and int(c.max()) > 0xFFFF:
                raise SystemExit("a (clause, quadrant) count exceeds u16")
            c = c.astype(np.uint16)
            del key, brk, starts, uk, cnt, uimg, ucol, dsel, ksel, p, img, o, q
        else:
            rows_img, ids, c = np.zeros(0, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint16)
        # rows of the kept images of this chunk, in ascending image order
        rank = np.full(i1 - i0, -1, np.int64)
        rank[kept] = np.arange(kept.shape[0])
        ip = np.zeros(kept.shape[0] + 1, np.int64)
        if rows_img.size:
            ip[1:] = np.cumsum(np.bincount(rank[rows_img - i0], minlength=kept.shape[0]))
        parts.append(F.SparseDocs(ip, ids, c, C * 4))
        del fw, fe, fj
    eng.synchronize()
    df_host = df.get()
    del df
    cp.get_default_memory_pool().free_all_blocks()
    wall = time.time() - t0
    return F.merge_rows(parts), {"wall_s": wall, "positions": n_pos, "fired_per_position": fired_total / max(1, n_pos),
                                 "positions_per_s": n_pos / max(1e-9, wall), "rows_kept": int(keep.sum())}, df_host


def masked_top1(eng, positions: np.ndarray, hidden: np.ndarray, chunk: int = 4096) -> float:
    cp = eng.cp
    hid = cp.asarray(hidden)
    tok = eng.corpus.tok
    hits = 0
    for lo in range(0, positions.shape[0], chunk):
        pos = positions[lo:lo + chunk]
        eng._sync_state_struct()
        fw = eng._forward(EVAL_EPOCH, cp.asarray(pos), hid, 0)
        eng._marks = []
        pred = cp.argmax(fw.scores, axis=1).get()  # first maximum: ties to the lower class id (docs/11 section 4)
        hits += int((pred == tok[pos.astype(np.int64)]).sum())
        del fw
    return hits / max(1, positions.shape[0])


def codes_check(eng, codes: F.SparseDocs, image_ids: np.ndarray, rs: np.random.Generator, n_check: int = 6) -> tuple[bool, bool]:
    """The GPU aggregation against the public fired_clauses (bruteforce on one position) and the CPU reference, on
    n_check random rows of `codes` (row r holds image image_ids[r])."""
    zero = np.zeros(int(eng.corpus.tok.shape[0]), np.uint8)
    ok_agg, ok_bf = True, True
    n_img = int(eng.corpus.doc_start.shape[0] - 1)
    for k, r in enumerate(np.sort(rs.choice(image_ids.shape[0], size=min(n_check, image_ids.shape[0]), replace=False)).tolist()):
        i = int(image_ids[r])
        pos = PT.patch_positions(np.asarray([i]))
        ev_p, ev_j = [], []
        for p in pos.tolist():
            a = eng.fired_clauses(EVAL_EPOCH, p, zero)
            if k == 0 and p == int(pos[24]):
                ok_bf &= np.array_equal(np.sort(eng.fired_clauses_bruteforce(EVAL_EPOCH, p, zero)), np.sort(a))
            ev_p.append(np.full(a.shape[0], p, np.int64))
            ev_j.append(a.astype(np.int64))
        ref = PT.image_code_from_events(np.concatenate(ev_p), np.concatenate(ev_j), np.zeros(n_img, np.int64), 1, int(eng.cfg.n_clauses))
        got = codes.rows(np.asarray([r]))
        ok_agg &= np.array_equal(ref.ids, got.ids) and np.array_equal(ref.cnt, got.cnt.astype(ref.cnt.dtype))
    return ok_agg, ok_bf


def save_codes(path: Path, sd: F.SparseDocs, **extra) -> str:
    np.savez(path, indptr=sd.indptr, ids=sd.ids, cnt=sd.cnt, n_cols=np.int64(sd.n_cols), **extra)
    return hashing.sha256_file(path)


def load_codes(path: Path) -> tuple[F.SparseDocs, np.ndarray, np.ndarray]:
    """(codes over the saved rows, image id of every row, pool document frequency of every column)."""
    z = np.load(path)
    return F.SparseDocs(z["indptr"], z["ids"], z["cnt"], int(z["n_cols"])), z["image_ids"], z["df_pool"]


def stage_gpu(cfg: dict, args) -> int:
    from tmlm.data.synthetic import sym_freq_of
    from tmlm.engine.api import CorpusArrays, state_hash_of
    from tmlm.engine.checkpoint import save_checkpoint
    from tmlm.engine.config import EngineConfig
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    P = cfg["params"]
    seed = int(args.seed if args.seed is not None else P["seed"])
    ca, extras, meta = PT.load_store(P["corpus"], int(P["n_codes"]) + 1, data_root(args), check=True)
    M("seed", seed)
    M("store_sha256", meta["sha256"])
    V = int(ca.kind_map.shape[1])
    n_img = int(ca.doc_start.shape[0] - 1)
    dev = extras["dev"].astype(bool)
    pool_img = np.flatnonzero(~dev)
    dev_img = np.flatnonzero(dev)
    tgt_all = np.asarray(ca.targets)
    if P["f1"]:
        keep = PT.f1_keep(tgt_all, ca.tok[tgt_all.astype(np.int64)], t=float(P["f1_t"]))
        targets = tgt_all[keep].astype(np.uint32)
    else:
        targets = tgt_all
    M("targets.all", int(tgt_all.shape[0]))
    M("targets.after_f1", int(targets.shape[0]))
    M("targets.sha256", hashing.sha256_array(targets))
    pool_tok = ca.tok.reshape(n_img, PT.TOKENS_PER_IMAGE)[pool_img].reshape(-1)
    ecfg = dict(P["engine"])
    ecfg.update({"mode": "masked", "vocab": V, "seed": seed})
    if args.clauses:
        ecfg["n_clauses"] = int(args.clauses)
    cfg_e = EngineConfig(**ecfg)
    M("engine_config", cfg_e.to_dict())
    epochs = int(args.epochs if args.epochs is not None else P["epochs"])
    hidden_eval = PT.frozen_hidden(int(ca.tok.shape[0]))
    dev_pos = PT.patch_positions(dev_img)
    gold = ca.tok[dev_pos.astype(np.int64)]
    uni = int(np.bincount(ca.tok[targets.astype(np.int64)].astype(np.int64), minlength=V).argmax())
    M("qpred.dev_positions", int(dev_pos.shape[0]))
    M("qpred.unigram_top1", float(np.mean(gold == uni)), note=f"most frequent training target {uni}")
    # rows kept in the code files: the dev images and every labelled pool image of this seed's draws (all images are
    # evaluated; the document frequency over the whole pool, for the TM probe's column choice, is kept as a vector)
    Q = P["probe"]
    draws = label_draws(extras["labels"].astype(np.int64), pool_img, seed, Q["budgets"], int(Q["draws"]))
    keep_rows = dev.copy()
    for ids in draws.values():
        keep_rows[ids] = True
    keep_ids = np.flatnonzero(keep_rows)
    M("codes.rows_kept", int(keep_ids.shape[0]), note="dev images plus the labelled pool images of every (budget, draw)")
    rs = np.random.default_rng(12345)
    for arm in args.arms.split(","):
        km = np.asarray(ca.kind_map) if arm == "main" else PT.random_kind_map(np.asarray(ca.kind_map), seed)
        sf = sym_freq_of(pool_tok, km, V)
        if arm == "main":
            check("main.sym_freq_equals_store", np.array_equal(sf, np.asarray(ca.sym_freq)))
        else:
            M(f"{arm}.kind_map_sha256", hashing.sha256_array(km))
        corpus = CorpusArrays(tok=np.asarray(ca.tok), doc_start=np.asarray(ca.doc_start), kind_map=km,
                              stop=np.asarray(ca.stop), sym_freq=sf, targets=targets)
        eng = CudaEngine(cfg_e, corpus)
        nb = eng.n_batches()
        young_b = max(1, int(math.ceil(float(P["young_fraction"]) * nb)))
        M(f"{arm}.batches_per_epoch", nb)
        banks: dict[str, dict] = {}
        for e in range(epochs):
            t0 = time.time()
            diag: dict[str, int] = {}
            for b in range(nb):
                d = eng.train_batch(e, b)
                for k, v in d.items():
                    diag[k] = diag.get(k, 0) + int(v)
                if arm == "main" and e == 0 and b + 1 == young_b:
                    eng.synchronize()
                    path = run_dir() / f"{arm}_young.npz"
                    h = save_checkpoint(path, eng, 0, b + 1, compress=True)
                    banks["young"] = {"state": eng.state(), "hash": h, "batches": b + 1}
                    M(f"{arm}.young.state_hash", h, batches=b + 1, file=path.name)
            eng.synchronize()
            wall = time.time() - t0
            st = eng.state()
            if e + 1 == epochs:  # only the final bank (and the young one) is kept as a file; every epoch's hash is logged
                h = save_checkpoint(run_dir() / f"{arm}_final.npz", eng, e + 1, 0, compress=True)
            else:
                h = state_hash_of(st, cfg_e)
            top1 = masked_top1(eng, dev_pos, hidden_eval)
            M(f"{arm}.epoch{e + 1}", {"wall_s": wall, "examples_per_s": diag.get("n_examples", 0) / max(1e-9, wall),
                                      "alive": int(st["alive"].sum()), "state_hash": h, "dev_masked_top1": top1, "diag": diag})
            log(f"{arm} epoch {e + 1}: {wall:.1f} s, alive {int(st['alive'].sum())}, dev masked top-1 {top1:.4f}")
            flush()
        banks["final"] = {"state": eng.state(), "hash": h, "epochs": epochs}
        for bank in (["final", "young"] if arm == "main" else ["final"]):
            eng.load_state(banks[bank]["state"])
            if bank == "young":
                M(f"{arm}.young.dev_masked_top1", masked_top1(eng, dev_pos, hidden_eval))
            codes, info, df_pool = image_codes(eng, n_img, int(P["chunk_images"]), keep=keep_rows, df_rows=~dev)
            ok_agg, ok_bf = codes_check(eng, codes, keep_ids, rs)
            check(f"{arm}.{bank}.codes_equal_cpu_reference", ok_agg)
            check(f"{arm}.{bank}.fired_equal_bruteforce", ok_bf)
            name = f"codes_{arm}_{bank}.npz"
            sha = save_codes(run_dir() / name, codes, image_ids=keep_ids, df_pool=df_pool, state_hash=banks[bank]["hash"])
            nnz = np.diff(codes.indptr)
            M(f"{arm}.{bank}.codes", {**info, "file": name, "sha256": sha, "nnz": int(codes.ids.shape[0]),
                                      "distinct_per_image_mean": float(nnz.mean()), "state_hash": banks[bank]["hash"],
                                      "columns_used": int(np.unique(codes.ids).shape[0])})
            log(f"{arm}.{bank}: codes {info['positions_per_s']:.0f} pos/s, fired/pos {info['fired_per_position']:.1f}, distinct/image {nnz.mean():.0f}")
            flush()
        del eng, banks
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ probes (CPU)
def representations(cfg: dict, args, gpu_dir: Path, ca, extras) -> dict:
    """name -> (matrix, dense flag, row_of: int64 [n_images] row of an image in the matrix, -1 if absent,
    df_pool: document frequency of every column over the pool images)."""
    P = cfg["params"]
    n_img = int(ca.doc_start.shape[0] - 1)
    pool = ~extras["dev"].astype(bool)
    reps = {}
    for rep, fname in P["probe"]["bank_codes"].items():
        f = gpu_dir / fname
        if f.exists():
            sd, image_ids, df_pool = load_codes(f)
            row_of = np.full(n_img, -1, np.int64)
            row_of[image_ids] = np.arange(image_ids.shape[0])
            reps[rep] = (sd.to_csr("log1p", l2=True), False, row_of, df_pool)
            M(f"rep.{rep}.file_sha256", hashing.sha256_file(f))
        else:
            M(f"rep.{rep}.missing", str(f))
    bovw = PT.bovw_quadrant(ca.tok, np.arange(n_img), vocab=int(ca.kind_map.shape[1]))
    Xb = bovw.to_csr("log1p", l2=True)
    reps["bovw"] = (Xb, False, np.arange(n_img), np.bincount(Xb[np.flatnonzero(pool)].indices, minlength=Xb.shape[1]))
    return reps


def take(rep: tuple, images: np.ndarray):
    X, _, row_of, _ = rep
    r = row_of[np.asarray(images, np.int64)]
    if np.any(r < 0):
        raise SystemExit("an image needed by a probe has no row in the code file (gpu and probe configs differ?)")
    return X[r]


def pixel_matrices(args, P: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grey / 255 float32 [n, 784], binarised uint8 [n, 784], labels) of the store's images, re-read from the parquet file
    and checked against the store's tokens (the store keeps tokens, not pixels)."""
    cache = Path(args.hf_cache) if args.hf_cache else HF_CACHE
    path = hub_file(cache, offline=True)
    images, labels = PT.read_fashion_parquet(path, limit=P.get("limit_images"))
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0, PT.binarize(images).reshape(images.shape[0], -1), labels


def tm_ctm_fit_predict(Xl, yl, Xe, cfg: dict, seed: int, chunk: int = 2000) -> np.ndarray:
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

    np.random.seed(int(seed))
    tm = TMCoalescedClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]), s=float(cfg["s"]), platform="CPU",
                               weighted_clauses=True, feature_negation=True, seed=int(seed))
    Xl32 = np.ascontiguousarray(Xl, dtype=np.uint32)
    yl32 = np.ascontiguousarray(yl, dtype=np.uint32)
    for _ in range(int(cfg["epochs"])):
        tm.fit(Xl32, yl32)
    out = [np.asarray(tm.predict(np.ascontiguousarray(Xe[i:i + chunk], dtype=np.uint32))) for i in range(0, Xe.shape[0], chunk)]
    return np.concatenate(out).astype(np.int64)


def tm_cv_cell(Xl, yl, Xe, ye, grid: list, seed: int, folds: int) -> dict:
    from tmlm.ssl import transfer as TR

    t0 = time.time()
    cv = {}
    splits = TR.cv_splits(yl, seed, folds)
    if len(grid) > 1:
        for gi, g in enumerate(grid):
            accs = [float(np.mean(tm_ctm_fit_predict(Xl[tr], yl[tr], Xl[va], g, seed * 1000 + fi) == yl[va])) for fi, (tr, va) in enumerate(splits)]
            cv[gi] = float(np.mean(accs))
        gi = min(g for g, a in cv.items() if a == max(cv.values()))
    else:
        gi = 0
    pred = tm_ctm_fit_predict(Xl, yl, Xe, grid[gi], seed)
    return {"acc": float(np.mean(pred == ye)), "choice": grid[gi], "cv": cv, "fit_s": time.time() - t0,
            "pred_sha256": hashing.sha256_bytes(pred.astype("<i8").tobytes())}


def gpu_seed(gpu_dir: Path) -> int:
    m = json.loads((gpu_dir / "metrics.json").read_text())
    return int(m["seed"]["value"])


def stage_probe(cfg: dict, args, tm_only: bool) -> int:
    from tmlm.ssl import transfer as TR

    P = cfg["params"]
    Q = P["probe"]
    gpu_dir = Path(args.gpu_run)
    seed = gpu_seed(gpu_dir)
    M("seed", seed)
    M("gpu_run", str(gpu_dir))
    ca, extras, meta = PT.load_store(P["corpus"], int(P["n_codes"]) + 1, data_root(args), check=True)
    labels = extras["labels"].astype(np.int64)
    dev = extras["dev"].astype(bool)
    pool_img, dev_img = np.flatnonzero(~dev), np.flatnonzero(dev)
    y_dev = labels[dev_img]
    reps = representations(cfg, args, gpu_dir, ca, extras)
    if not tm_only:
        grey, binpix, lab2 = pixel_matrices(args, P)
        check("pixels_labels_equal_store_labels", np.array_equal(lab2.astype(np.int64), labels))
        tok_check = PT.image_tokens(PT.patch_patterns(binpix.reshape(-1, 28, 28)), extras["lut"], int(P["n_codes"]))
        check("pixels_tokenize_to_store_tokens", np.array_equal(tok_check, np.asarray(ca.tok)))
        ident = np.arange(labels.shape[0])
        reps["pixels"] = (grey, True, ident, None)
        reps["pixels_bin"] = (binpix.astype(np.float32), True, ident, None)
    else:
        binpix = None
    flush()
    draws = label_draws(labels, pool_img, seed, Q["budgets"], int(Q["draws"]))
    for budget in Q["budgets"]:
        for draw in range(int(Q["draws"])):
            lab = draws[(int(budget), draw)]
            yl = labels[lab]
            base = {"seed": seed, "budget": int(budget), "draw": draw, "labelled_sha256": hashing.sha256_bytes(lab.astype("<i8").tobytes())}
            if not tm_only:
                for rep, R in reps.items():
                    r = TR.lr_cell(take(R, lab), yl, take(R, dev_img), y_dev, seed=draw, dense=R[1], max_iter=int(Q["lr_max_iter"]),
                                   c_grid=tuple(Q["c_grid"]), max_folds=int(Q["folds"]))
                    write_cell({**base, "rep": rep, "probe": "lr", "acc": r.accuracy, "choice": r.choice, "cv": r.cv, "fit_s": r.fit_s,
                                "pred_sha256": hashing.sha256_bytes(np.asarray(r.pred, np.int64).astype("<i8").tobytes())})
                    log(f"budget {budget} draw {draw} {rep} lr: {r.accuracy:.4f} (C {r.choice}, {r.fit_s:.1f} s)")
                c = tm_cv_cell(binpix[lab], yl, binpix[dev_img], y_dev, Q["tm_supervised_grid"], seed * 100 + draw, int(Q["tm_folds"]))
                write_cell({**base, "rep": "pixels_bin", "probe": "tm_supervised", **c})
                log(f"budget {budget} draw {draw} supervised tmu CTM on binarised pixels: {c['acc']:.4f} ({c['fit_s']:.0f} s)")
            else:
                for rep in Q["tm_probe_reps"]:
                    if rep not in reps:
                        continue
                    R = reps[rep]
                    cols = F.top_df_columns(R[3], int(Q["tm_probe_k"]))
                    Xl = (take(R, lab)[:, cols] > 0).astype(np.uint8).toarray()
                    Xe = (take(R, dev_img)[:, cols] > 0).astype(np.uint8).toarray()
                    c = tm_cv_cell(Xl, yl, Xe, y_dev, [Q["tm_probe_cfg"]], seed * 100 + draw, 3)
                    write_cell({**base, "rep": rep, "probe": "tm_code", "k": int(cols.shape[0]), **c})
                    log(f"budget {budget} draw {draw} {rep} tm probe: {c['acc']:.4f} ({c['fit_s']:.0f} s)")
            flush()
    M("wall_s", time.time() - T0)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ report
def paired_bootstrap(d: np.ndarray, n_boot: int = 10000, seed: int = 20260924) -> dict:
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.shape[0], size=(n_boot, d.shape[0]))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)), "n": int(d.shape[0])}


def h17_10(cells: list[dict], budget: int, main: str = "code_main", refs=("code_young", "bovw"), margin: float = 0.02) -> dict:
    """code_main minus the stronger (higher mean over all its (seed, draw) cells) of refs, paired over (seed, draw)."""
    from tmlm.ssl import transfer as TR

    acc = {}
    for c in cells:
        if c["probe"] == "lr" and c["budget"] == budget:
            acc[(c["rep"], c["seed"], c["draw"])] = c["acc"]
    keys = sorted({(s, d) for (_, s, d) in acc})
    means = {r: float(np.mean([acc[(r, s, d)] for (s, d) in keys if (r, s, d) in acc])) for r in refs if any((r, s, d) in acc for (s, d) in keys)}
    if main not in {r for (r, _, _) in acc} or not means:
        return {"status": "incomplete"}
    ref = max(means, key=lambda r: (means[r], -refs.index(r)))
    pairs = [(s, d) for (s, d) in keys if (main, s, d) in acc and (ref, s, d) in acc]
    diff = np.asarray([acc[(main, s, d)] - acc[(ref, s, d)] for (s, d) in pairs])
    boot = paired_bootstrap(diff)
    return {"budget": budget, "reference": ref, "reference_means": means, "pairs": len(pairs),
            "main_mean": float(np.mean([acc[(main, s, d)] for (s, d) in pairs])), "diff_bootstrap": boot,
            "diff_t": TR.t_interval(list(diff)), "meets": bool(boot["mean"] >= margin and boot["lo"] > 0)}


def stage_report(cfg: dict, args) -> int:
    from tmlm.ssl import transfer as TR

    cells, gpu_metrics = [], {}
    for d in args.runs.split(","):
        d = Path(d)
        if (d / "cells.jsonl").exists():
            cells += [json.loads(line) for line in (d / "cells.jsonl").read_text().splitlines() if line.strip()]
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            if "main.final.codes" in m:
                gpu_metrics[str(m["seed"]["value"])] = {k: v["value"] for k, v in m.items() if k.startswith(("main.", "randkm.", "qpred."))}
    table = {}
    for c in cells:
        key = f"{c['probe']}.{c['rep']}.b{c['budget']}"
        table.setdefault(key, []).append(c["acc"])
    M("table", {k: {**TR.t_interval(v), "min": min(v), "max": max(v)} for k, v in sorted(table.items())}, label="derived")
    M("seeds_in_cells", sorted({c["seed"] for c in cells}), label="derived")
    V = cfg["params"]["verdict"]
    res = {f"b{b}": h17_10(cells, int(b), margin=float(V["margin"])) for b in cfg["params"]["probe"]["budgets"]}
    M("verdict.H17_10", res, label="derived", rule=V["rule"], primary_budget=V["budget"])
    M("gpu_runs", gpu_metrics, label="derived")
    flush()
    log(json.dumps(res, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "gpu", "probe", "probe_tm", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="override the data root (smoke: a scratch directory)")
    ap.add_argument("--hf-cache", default=None, help="override the Hugging Face cache directory (smoke: scratch)")
    ap.add_argument("--write-prereg-hash", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arms", default="main,randkm")
    ap.add_argument("--clauses", type=int, default=None, help="override n_clauses (smoke)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke)")
    ap.add_argument("--gpu-run", default=None)
    ap.add_argument("--runs", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    M("config_file", args.config, sha256=hashing.sha256_file(args.config))
    M("stage", args.stage)
    try:
        if args.stage == "data":
            return stage_data(cfg, args)
        if args.stage == "gpu":
            return stage_gpu(cfg, args)
        if args.stage in ("probe", "probe_tm"):
            return stage_probe(cfg, args, tm_only=args.stage == "probe_tm")
        return stage_report(cfg, args)
    finally:
        flush()


if __name__ == "__main__":
    sys.exit(main())
