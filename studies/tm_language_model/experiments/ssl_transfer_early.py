"""EARLY TRANSFER TEST of self-supervised clause banks (EXPLORATORY, docs/12 F23; not pre-registered, no
docs/13 threshold is tested; downstream train and validation/dev data only, no downstream test split is
downloaded or read, and the WikiText-103 test split is never loaded).

Question: are the clauses that Tsetlin automata learned self-supervised (causal next-token feedback on
WikiText-103) useful LOGICAL REPRESENTATIONS of text beyond language modelling? A text is represented by
the clauses of a FROZEN bank that fire on it (every position's fired set of the left context, summed
over the positions of the text: src/tmlm/ssl/features.py), and a classifier is trained on top with few
labels. Controls: the YOUNG bank (same engine and config, 7.9 percent of one epoch of feedback: the
untrained control C0), bag of BPE tokens, in-house CBOW means, exchange-cluster bags.

Stages (each one a logged run; configs in experiments/cvtm/configs/ssl_transfer_early_*.json):
  data      download the pinned parquet files (train and validation/dev only) into data/hf_cache/hub,
            sha256 them against the hub's LFS oid, carve the dev splits of AG News and TREC out of
            train, and write docs/notes/preregistration-data-hashes-A6-<dataset>.json once (write-once;
            a later run re-reads and verifies it).
  features  (GPU shared) tokenize every text with the WikiText-103 BPE-8k tokenizer, build ONE
            CorpusArrays of all documents (kind_map, stop, sym_freq of WikiText-103 train), and for each
            bank (CP anchor epoch 4, epoch 1, young) run CudaEngine._forward (the batched forward of
            predict_scores; read-only) over every position, aggregate the fired clause ids per document
            on the GPU, and save the CSR (features_<bank>.npz). Pipeline checks against the public
            fired_clauses / fired_clauses_bruteforce and the CPU reference aggregation.
  classify  (CPU) label budgets 64, 256, 1024, 4096 and full, seeds 101, 102, 103; L2 logistic
            regression (C by CV on the labelled sample) and/or the tmu Tsetlin machine (grid by CV on the
            labelled sample); accuracy on the validation/dev split. One JSON line per cell (cells.jsonl).
  report    merge the cells of classify runs, t-intervals over seeds, paired differences, EARLY VERDICT.

    source env/env.sh
    scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py \\
        --name ssl_transfer_data_exploratory --config experiments/cvtm/configs/ssl_transfer_early_data.json -- \\
        .venv-tm/bin/python -u experiments/ssl_transfer_early.py --stage data
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import transfer as TR  # noqa: E402

T0 = time.time()
METRICS: dict = {}
FAILED: list = []
HF_CACHE = ROOT / "data" / "hf_cache" / "hub"
CP_DIR = ROOT / "results" / "20260923T042350Z_CP_anchor"
YOUNG = ROOT / "results" / "20260924T022628Z_ckm_probe_causal_controls_exploratory" / "young_bank_checkpoint.npz"
CBOW = ROOT / "results" / "20260922T210329Z_M2_baselines_cbow_knn_v8000" / "cbow_w_in.npy"
BANKS = {
    "cp_e4": {"path": CP_DIR / "checkpoint_epoch4.npz", "logged": (CP_DIR / "metrics.json", "epoch4.checkpoint")},
    "cp_e1": {"path": CP_DIR / "checkpoint_epoch1.npz", "logged": (CP_DIR / "metrics.json", "epoch1.checkpoint")},
    "young": {"path": YOUNG, "logged": (YOUNG.parent / "metrics.json", "young.checkpoint")},
}
EPOCH = 0          # causal mode: tm_hidden is 0 everywhere, the epoch has no effect on the forward pass
VOCAB = 8000
TASKS = {
    # code, repo, revision, files (split -> path in repo), text columns, label column, licence of the card
    "sst2": {"code": 1, "repo": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
             "files": {"train": "sst2/train-00000-of-00001.parquet", "validation": "sst2/validation-00000-of-00001.parquet"},
             "text": ["sentence"], "label": "label", "n_classes": 2, "dev_from_train": 0,
             "card_license": "other (GLUE card: 'We refer users to the original licenses accompanying each dataset')"},
    "agnews": {"code": 2, "repo": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
               "files": {"train": "data/train-00000-of-00001.parquet"},
               "text": ["text"], "label": "label", "n_classes": 4, "dev_from_train": 7600,
               "card_license": "unknown (dataset card)"},
    "trec": {"code": 3, "repo": "CogComp/trec", "revision": "65752bf53af25bc935a0dce92fb5b6c930728450",
             "files": {"train": "default/train/0000.parquet"},
             "text": ["text"], "label": "coarse_label", "n_classes": 6, "dev_from_train": 500,
             "card_license": "unknown (dataset card)"},
}
HASH_FILE = ROOT / "docs" / "notes" / "preregistration-data-hashes-A6-{task}.json"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}


def flush() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        runlog.write_metrics(METRICS, on_conflict="replace")


def check(name: str, ok: bool, **extra) -> None:
    M(f"check.{name}", bool(ok), **extra)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {extra}")


def run_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_transfer_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def wait_for_headroom(peak_gib: float, what: str, frac: float = 0.80, poll_s: float = 10.0, max_wait_s: float = 5400.0) -> float:
    """Block until cgroup memory.current + peak_gib < frac * memory.max (the task's headroom rule); returns
    the seconds waited. Raises after max_wait_s."""
    t0 = time.time()
    mx = int(Path("/sys/fs/cgroup/memory.max").read_text())
    said = False
    while True:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if cur + peak_gib * 2 ** 30 < frac * mx:
            w = time.time() - t0
            log(f"memory before {what}: current {cur / 2 ** 30:.1f} GiB + peak {peak_gib:.0f} GiB < {frac:.2f} x {mx / 2 ** 30:.0f} GiB (waited {w:.0f} s)")
            return w
        if not said:
            log(f"memory before {what}: current {cur / 2 ** 30:.1f} GiB + peak {peak_gib:.0f} GiB >= {frac:.2f} x max: waiting")
            said = True
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"no memory headroom for {what} after {max_wait_s:.0f} s")
        time.sleep(poll_s)


def mem_gib() -> dict:
    cur = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2 ** 30
    return {"rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "cgroup_current_gib": cur}


# ------------------------------------------------------------------ data
def hub_file(task: str, rel: str, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download

    t = TASKS[task]
    return Path(hf_hub_download(t["repo"], rel, repo_type="dataset", revision=t["revision"], cache_dir=str(HF_CACHE),
                                local_files_only=offline))


def hub_lfs_sha(task: str) -> dict:
    """path -> (LFS sha256 or None, git blob id). A file of a refs/convert/parquet commit can be a plain git
    blob (no LFS record): its integrity id is then the git blob sha1 (checked by git_blob_sha1)."""
    from huggingface_hub import HfApi

    t = TASKS[task]
    out = {}
    for e in HfApi().list_repo_tree(t["repo"], repo_type="dataset", revision=t["revision"], recursive=True, expand=True):
        if str(e.path) in t["files"].values():
            lfs = getattr(e, "lfs", None)
            sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs is not None else None
            out[str(e.path)] = (sha, getattr(e, "blob_id", None))
    return out


def git_blob_sha1(path: Path) -> str:
    import hashlib

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_rows(task: str, split_file: Path) -> tuple[list[str], np.ndarray]:
    from datasets import Dataset

    t = TASKS[task]
    ds = Dataset.from_parquet(str(split_file), cache_dir=str(ROOT / "data" / "hf_cache" / "datasets"))
    cols = ds.to_dict()
    texts = [" ".join(str(cols[c][i]) for c in t["text"]) for i in range(len(ds))]
    return texts, np.asarray(cols[t["label"]], np.int64)


def dev_carve(y: np.ndarray, n_dev: int, task_code: int) -> np.ndarray:
    """bool [n]: the dev part carved out of train, stratified: class c gets round(n_dev * share_c) examples
    (largest remainders fix the total), chosen by a seeded permutation inside the class."""
    classes, counts = np.unique(y, return_counts=True)
    share = counts / counts.sum() * n_dev
    k = np.floor(share).astype(np.int64)
    rem = n_dev - int(k.sum())
    k[np.argsort(-(share - k), kind="stable")[:rem]] += 1
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 622, int(task_code)]))
    dev = np.zeros(y.shape[0], bool)
    for c, kc in zip(classes, k):
        idx = np.flatnonzero(y == c)
        dev[idx[rng.permutation(idx.shape[0])[:kc]]] = True
    return dev


def load_task(task: str, offline: bool = True) -> dict:
    """pool and dev texts and labels of a task (validation/dev only; test never downloaded)."""
    t = TASKS[task]
    tr_texts, tr_y = read_rows(task, hub_file(task, t["files"]["train"], offline))
    if t["dev_from_train"]:
        dev = dev_carve(tr_y, t["dev_from_train"], t["code"])
        pool_idx, dev_idx = np.flatnonzero(~dev), np.flatnonzero(dev)
        return {"pool_text": [tr_texts[i] for i in pool_idx], "pool_y": tr_y[pool_idx],
                "dev_text": [tr_texts[i] for i in dev_idx], "dev_y": tr_y[dev_idx],
                "pool_src_idx": pool_idx, "dev_src_idx": dev_idx, "dev_source": "carved from train"}
    va_texts, va_y = read_rows(task, hub_file(task, t["files"]["validation"], offline))
    return {"pool_text": tr_texts, "pool_y": tr_y, "dev_text": va_texts, "dev_y": va_y,
            "pool_src_idx": np.arange(len(tr_texts)), "dev_src_idx": np.arange(len(va_texts)), "dev_source": "validation split"}


def idx_sha(a: np.ndarray) -> str:
    return hashing.sha256_bytes(np.asarray(a, np.int64).astype("<i8").tobytes())


def stage_data(args) -> int:
    from tmlm.data import bpe

    for task, t in TASKS.items():
        files = {}
        lfs = hub_lfs_sha(task)
        for split, rel in t["files"].items():
            p = hub_file(task, rel, offline=False)
            sha = hashing.sha256_file(p)
            lfs_sha, blob_id = lfs.get(rel, (None, None))
            files[rel] = {"sha256": sha, "bytes": int(p.stat().st_size), "split": split,
                          "hub_lfs_sha256": lfs_sha, "hub_lfs_sha256_equal": bool(lfs_sha == sha)}
            if lfs_sha is not None:
                check(f"{task}.{split}.sha256_equals_hub_lfs_oid", lfs_sha == sha, sha256=sha, hub=lfs_sha)
            else:
                g = git_blob_sha1(p)
                check(f"{task}.{split}.git_blob_sha1_equals_hub_blob_id (file is not LFS)", g == blob_id, git_blob_sha1=g, hub_blob_id=blob_id)
        d = load_task(task, offline=True)
        M(f"{task}.pool_size", len(d["pool_text"]))
        M(f"{task}.dev_size", len(d["dev_text"]))
        M(f"{task}.pool_class_counts", np.bincount(d["pool_y"], minlength=t["n_classes"]).tolist())
        M(f"{task}.dev_class_counts", np.bincount(d["dev_y"], minlength=t["n_classes"]).tolist())
        record = {
            "purpose": "EXPLORATORY early transfer test (docs/12 F23): downstream data hashed before any use; train and validation/dev only, no test split downloaded",
            "written_by": "experiments/ssl_transfer_early.py --stage data; write-once, never edited",
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {"repo": t["repo"], "repo_type": "dataset", "revision": t["revision"],
                       "license_dataset_card": t["card_license"],
                       "access": "huggingface_hub.hf_hub_download of the named files only into data/hf_cache/hub, read with datasets.Dataset.from_parquet"},
            "files": files,
            "splits": {"pool": {"n": len(d["pool_text"]), "source_rows_sha256": idx_sha(d["pool_src_idx"])},
                       "dev": {"n": len(d["dev_text"]), "source": d["dev_source"], "source_rows_sha256": idx_sha(d["dev_src_idx"])},
                       "dev_rule": ("experiments/ssl_transfer_early.py dev_carve: stratified, class c gets round(n_dev * share_c) (largest remainders), "
                                    f"chosen by numpy default_rng(SeedSequence([20260924, 622, {t['code']}])).permutation inside each class; "
                                    "source_rows_sha256 = sha256 of the int64 little-endian row indices into the train file")
                       if t["dev_from_train"] else "the validation split of the file as is",
                       "row_index_format": "int64 little endian, ascending"},
            "text_columns": t["text"], "label_column": t["label"], "n_classes": t["n_classes"],
            "preprocessing": "src/tmlm/ssl/features.py doc_text: wikitext_style (as tmlm.cvtm.crtm.wikitext_style) plus ' \\n' (END token), BPE 8000 of data/tokenizers/bpe_8000",
            "tokenizer_sha256": bpe.tokenizer_sha256(VOCAB),
        }
        path = Path(str(HASH_FILE).format(task=task))
        if path.exists():
            old = json.loads(path.read_text())
            same = old.get("files", {}).keys() == record["files"].keys() and all(
                old["files"][k]["sha256"] == record["files"][k]["sha256"] for k in record["files"])
            check(f"{task}.hash_file_matches", same and old["splits"]["dev"]["source_rows_sha256"] == record["splits"]["dev"]["source_rows_sha256"])
            M(f"{task}.hash_file", "existing file read and verified; not rewritten")
        else:
            with open(path, "x", encoding="utf-8") as fh:          # exclusive create: never overwrites
                json.dump(record, fh, indent=1, sort_keys=True)
                fh.write("\n")
            M(f"{task}.hash_file", f"written once: {path.relative_to(ROOT)}", sha256=hashing.sha256_file(path))
        if runlog.current_run_dir() is not None:
            runlog.write_dataset_hashes({f"{task}_{r['split']}": r["sha256"] for r in files.values()}, on_conflict="replace")
        log(f"{task}: pool {len(d['pool_text'])}, dev {len(d['dev_text'])}")
    flush()
    return 1 if FAILED else 0


def verify_hash_files() -> dict:
    """Re-hash the local parquet files against the write-once hash files (features and classify stages)."""
    out = {}
    for task, t in TASKS.items():
        rec = json.loads(Path(str(HASH_FILE).format(task=task)).read_text())
        for rel, r in rec["files"].items():
            sha = hashing.sha256_file(hub_file(task, rel, offline=True))
            check(f"{task}.{r['split']}.sha256_equals_hash_file", sha == r["sha256"])
            out[f"{task}_{r['split']}"] = sha
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes(out, on_conflict="replace")
    return out


def all_docs(tasks: list[str], smoke: bool) -> tuple[dict, "F.TokenDocs", dict]:
    """Every pool and dev document of the tasks, tokenized, in the order task by task, pool then dev."""
    from tmlm.data import bpe

    tok = bpe.load_tokenizer(VOCAB)
    data, parts, layout = {}, [], {}
    for task in tasks:
        d = load_task(task, offline=True)
        if smoke:
            d = {k: (v[:600] if isinstance(v, (list, np.ndarray)) else v) for k, v in d.items()}
        data[task] = d
        for split in ("pool", "dev"):
            parts.append(F.encode(tok, d[f"{split}_text"]))
            layout[(task, split)] = len(parts) - 1
    docs, off = F.concat(parts)
    rows = {key: (int(off[i]), int(off[i + 1])) for key, i in layout.items()}
    return data, docs, rows


# ------------------------------------------------------------------ features (GPU)
def load_bank(name: str):
    from tmlm.engine.checkpoint import load_checkpoint

    b = BANKS[name]
    cfg, st, meta = load_checkpoint(b["path"])
    logged = json.loads(b["logged"][0].read_text())[b["logged"][1]]["value"]["state_hash"]
    check(f"bank.{name}.state_hash_equals_logged", meta["state_hash"] == logged, state_hash=meta["state_hash"])
    if cfg.mode != "causal" or int(cfg.vocab) != VOCAB:
        raise SystemExit(f"bank {name}: expected a causal V {VOCAB} bank")
    return cfg, st, meta


def doc_chunks(doc_start: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Consecutive document ranges [d0, d1) of about `target` positions (a longer document is alone)."""
    ds = doc_start.astype(np.int64)
    out, d0 = [], 0
    n_doc = ds.shape[0] - 1
    while d0 < n_doc:
        d1 = int(np.searchsorted(ds, ds[d0] + target, side="right")) - 1
        d1 = max(d0 + 1, min(d1, n_doc))
        out.append((d0, d1))
        d0 = d1
    return out


def bank_features(eng, docs: "F.TokenDocs", chunk: int) -> tuple["F.SparseDocs", dict]:
    cp = eng.cp
    ds = docs.doc_start.astype(np.int64)
    dop_dev = cp.asarray(docs.doc_of_pos().astype(np.int32))
    C = int(eng.cfg.n_clauses)
    parts, fired_total, cand_total = [], 0, 0
    fired_per_pos = np.zeros(docs.n_tok, np.int32)
    t0 = time.time()
    chunks = doc_chunks(docs.doc_start, chunk)
    step = max(1, len(chunks) // 10)
    for ci, (d0, d1) in enumerate(chunks):
        lo, hi = int(ds[d0]), int(ds[d1])
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, cp.arange(lo, hi, dtype=cp.uint32), None, 0)
        eng._marks = []
        cand_total += int(fw.n_candidates)
        fe, fj = fw.fired_ex, fw.fired_j
        n_f = int(fe.size)
        fired_total += n_f
        if n_f:
            fired_per_pos[lo:hi] = cp.bincount(fe.astype(cp.int64), minlength=hi - lo).get().astype(np.int32)
            drel = (dop_dev[fe.astype(cp.int64) + lo] - d0).astype(cp.uint64)
            key = (drel << cp.uint64(32)) | fj.astype(cp.uint64)
            key.sort()
            brk = cp.flatnonzero(key[1:] != key[:-1]) + 1
            starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
            uk = key[starts]
            cnt = cp.diff(cp.concatenate([starts, cp.asarray([n_f], dtype=starts.dtype)]))
            ukh = uk.get()
            dd = (ukh >> np.uint64(32)).astype(np.int64)
            ids = (ukh & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            c = cnt.get()
            if c.size and int(c.max()) > 65535:
                raise SystemExit("a per-document clause count exceeds u16")
            c = c.astype(np.uint16)
            del key, brk, starts, uk, cnt, drel
        else:
            dd = np.zeros(0, np.int64)
            ids = np.zeros(0, np.uint32)
            c = np.zeros(0, np.uint16)
        ip = np.zeros(d1 - d0 + 1, np.int64)
        ip[1:] = np.cumsum(np.bincount(dd, minlength=d1 - d0))
        parts.append(F.SparseDocs(ip, ids, c, C))
        del fw, fe, fj
        if (ci + 1) % step == 0:
            el = time.time() - t0
            log(f"  {hi}/{docs.n_tok} positions, {hi / el:.0f} pos/s, fired/pos {fired_total / hi:.1f}")
    eng.synchronize()
    wall = time.time() - t0
    cp.get_default_memory_pool().free_all_blocks()
    sd = F.merge_rows(parts)
    return sd, {"wall_s": wall, "pos_per_s": docs.n_tok / wall, "fired_per_position": fired_total / docs.n_tok,
                "candidates_per_position": cand_total / docs.n_tok, "fired_per_pos": fired_per_pos}


def stage_features(args) -> int:
    import cupy as cp
    from tmlm.data.corpus_store import load_split
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    verify_hash_files()
    tasks = args.tasks.split(",")
    data, docs, rows = all_docs(tasks, args.smoke)
    out = run_dir()
    ref = load_split("wikitext103", VOCAB, "train")
    corpus = F.corpus_arrays(docs, ref)
    np.savez(out / "tokens.npz", tok=docs.tok, doc_start=docs.doc_start, end_id=np.int64(docs.end_id),
             rows=json.dumps({f"{k[0]}.{k[1]}": v for k, v in rows.items()}),
             labels=json.dumps({t: {"pool": data[t]["pool_y"].tolist(), "dev": data[t]["dev_y"].tolist()} for t in tasks}))
    M("tokens.n_docs", docs.n_doc)
    M("tokens.n_tokens", docs.n_tok, note="text tokens plus one END token per document")
    for (t, s), (a, b) in rows.items():
        n = np.diff(docs.doc_start[a:b + 1].astype(np.int64))
        M(f"tokens.{t}.{s}.docs", int(b - a))
        M(f"tokens.{t}.{s}.tokens_per_doc_mean", float(n.mean()), note="including END")
    M("tokens.unknown_bytes_note", "byte-level BPE: every text is representable, no unknown tokens", label="derived")
    flush()
    rs = np.random.default_rng(12345)
    for name in args.banks.split(","):
        M(f"bank.{name}.memory_wait_s", wait_for_headroom(float(args.peak_gib), f"bank {name}"))
        t0 = time.time()
        cfg, st, meta = load_bank(name)
        eng = CudaEngine(cfg, corpus)
        eng.load_state(st)
        eng.synchronize()
        M(f"bank.{name}.engine_build_s", time.time() - t0)
        M(f"bank.{name}.alive", int(np.asarray(st["alive"]).sum()))
        log(f"bank {name}: forward over {docs.n_tok} positions")
        sd, info = bank_features(eng, docs, args.chunk)
        fpp = info.pop("fired_per_pos")
        for k, v in info.items():
            M(f"bank.{name}.{k}", v)
        # pipeline checks: our fired lists (via counts) vs the public API, aggregation vs the CPU reference
        chk_docs = np.sort(rs.choice(docs.n_doc, size=min(24, docs.n_doc), replace=False))
        ok_api, ok_bf, ok_agg = True, True, True
        for i, d in enumerate(chk_docs.tolist()):
            lo, hi = int(docs.doc_start[d]), int(docs.doc_start[d + 1])
            ev_d, ev_j = [], []
            for p in range(lo, hi):
                a = eng.fired_clauses(EPOCH, p)
                ok_api &= int(a.shape[0]) == int(fpp[p])
                if i < 2 and p == lo + (hi - lo) // 2:
                    b = eng.fired_clauses_bruteforce(EPOCH, p)
                    ok_bf &= np.array_equal(np.sort(b), np.sort(a))
                ev_d.append(np.zeros(a.shape[0], np.int64))
                ev_j.append(a.astype(np.int64))
            refd = F.aggregate_doc_counts(np.concatenate(ev_d), np.concatenate(ev_j), 1, sd.n_cols)
            got = sd.rows(np.asarray([d]))
            ok_agg &= np.array_equal(refd.ids, got.ids) and np.array_equal(refd.cnt, got.cnt)
        check(f"bank.{name}.fired_counts_equal_public_fired_clauses", ok_api, docs=len(chk_docs))
        check(f"bank.{name}.fired_equal_bruteforce", ok_bf, positions=2)
        check(f"bank.{name}.doc_aggregate_equals_cpu_reference", ok_agg, docs=len(chk_docs))
        nnz = np.diff(sd.indptr)
        M(f"bank.{name}.distinct_clauses_per_doc_mean", float(nnz.mean()))
        M(f"bank.{name}.distinct_clauses_per_doc_max", int(nnz.max()))
        M(f"bank.{name}.nnz", int(sd.ids.shape[0]))
        M(f"bank.{name}.max_count", int(sd.cnt.max()) if sd.cnt.size else 0)
        M(f"bank.{name}.clauses_firing_anywhere", int(np.unique(sd.ids).shape[0]))
        for (t, s), (a, b) in rows.items():
            M(f"bank.{name}.{t}.{s}.distinct_clauses_per_doc_mean", float(nnz[a:b].mean()))
        np.savez(out / f"features_{name}.npz", indptr=sd.indptr, ids=sd.ids, cnt=sd.cnt, n_cols=np.int64(sd.n_cols),
                 state_hash=meta["state_hash"])
        M(f"bank.{name}.features_file", f"features_{name}.npz", sha256=hashing.sha256_file(out / f"features_{name}.npz"))
        M(f"memory.after_{name}", mem_gib())
        log(f"bank {name}: {info['pos_per_s']:.0f} pos/s, fired/pos {info['fired_per_position']:.1f}, "
            f"distinct/doc {nnz.mean():.0f}, nnz {sd.ids.shape[0]} ({time.time() - t0:.0f} s)")
        del eng, st, sd
        cp.get_default_memory_pool().free_all_blocks()
        flush()
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ classify (CPU)
class FeatureStore:
    """All representations of a features run, per task: build once, slice rows per cell."""

    def __init__(self, feat_dir: Path, tasks: list[str], k_top: int, cbow_bits: int) -> None:
        z = np.load(feat_dir / "tokens.npz")
        self.docs = F.TokenDocs(tok=z["tok"], doc_start=z["doc_start"], end_id=int(z["end_id"]))
        self.rows = {tuple(k.split(".")): tuple(v) for k, v in json.loads(str(z["rows"])).items()}
        self.labels = json.loads(str(z["labels"]))
        self.feat_dir = feat_dir
        self.tasks = tasks
        self.k_top = k_top
        self.cbow_bits = cbow_bits
        self._bank: dict = {}
        self._base: dict = {}

    def task_rows(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.rows[(task, "pool")]
        c, d = self.rows[(task, "dev")]
        return np.arange(a, b), np.arange(c, d)

    def y(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.labels[task]["pool"], np.int64), np.asarray(self.labels[task]["dev"], np.int64)

    def bank(self, name: str) -> "F.SparseDocs":
        if name not in self._bank:
            z = np.load(self.feat_dir / f"features_{name}.npz")
            self._bank[name] = F.SparseDocs(z["indptr"], z["ids"], z["cnt"], int(z["n_cols"]))
        return self._bank[name]

    def base(self, kind: str) -> "F.SparseDocs":
        if kind not in self._base:
            if kind == "bpe_uni":
                self._base[kind] = F.bag_of_tokens(self.docs, VOCAB, bigrams=False)
            elif kind == "bpe_unibi":
                self._base[kind] = F.compact_columns(F.bag_of_tokens(self.docs, VOCAB, bigrams=True))[0]
            elif kind == "clusters":
                from tmlm.data.corpus_store import load_split
                km = np.asarray(load_split("wikitext103", VOCAB, "train").kind_map)
                self._base[kind] = F.cluster_bags(self.docs, km)[0]
            elif kind == "cbow":
                self._base[kind] = F.mean_embedding(self.docs, np.load(CBOW))
            else:
                raise KeyError(kind)
        return self._base[kind]

    def source(self, fs: str):
        """feature set name -> (sparse docs or dense matrix, is_dense)."""
        if fs.startswith("cbow"):
            return self.base("cbow"), True
        for b in BANKS:
            if fs.startswith(b + "."):
                return self.bank(b), False
        return self.base(fs.split(".")[0]), False

    def matrices(self, fs: str, task: str, lab_rows: np.ndarray, dev_rows: np.ndarray, pool_rows: np.ndarray, for_tm: bool):
        """(X_lab, X_dev, info). Variants: '<src>.log1p' (all columns, log(1 + count), unit rows),
        '<src>.top' (the k_top columns of highest DF over the unlabelled POOL, binary, unit rows for lr,
        raw binary uint32 for tm), 'cbow' (dense means; tm: thermometer bits from labelled-sample quantiles)."""
        src, dense = self.source(fs)
        if dense:
            Xl, Xd = src[lab_rows], src[dev_rows]
            if for_tm:
                edges = F.quantile_edges(Xl, self.cbow_bits)
                return F.thermometer(Xl, edges), F.thermometer(Xd, edges), {"dim": int(Xl.shape[1] * self.cbow_bits)}
            return Xl, Xd, {"dim": int(Xl.shape[1])}
        variant = fs.split(".")[-1]
        if variant == "log1p" and not for_tm:
            sub = src.rows(np.concatenate([lab_rows, dev_rows]))
            X = sub.to_csr("log1p")
            n = lab_rows.shape[0]
            return X[:n], X[n:], {"dim": int(src.n_cols)}
        if variant == "top":
            key = (fs, task)
            if key not in self._base:
                assert np.array_equal(pool_rows, np.arange(pool_rows[0], pool_rows[-1] + 1))
                self._base[key] = F.top_df_columns(src.doc_freq_range(int(pool_rows[0]), int(pool_rows[-1]) + 1), self.k_top)
            cols = self._base[key]
            sub = src.rows(np.concatenate([lab_rows, dev_rows]))
            X = sub.to_csr("binary", cols=cols, l2=not for_tm)
            n = lab_rows.shape[0]
            if for_tm:
                Xd = np.asarray(X.toarray(), dtype=np.uint32)
                return Xd[:n], Xd[n:], {"dim": int(cols.shape[0])}
            return X[:n], X[n:], {"dim": int(cols.shape[0])}
        raise ValueError(f"feature set {fs} not available for {'tm' if for_tm else 'lr'}")


def stage_classify(args) -> int:
    cfg = json.loads(Path(args.config).read_text())
    p = cfg["params"]
    feat_dir = ROOT / (args.features_run or p["features_run"])
    verify_hash_files()
    store = FeatureStore(feat_dir, p["tasks"], int(p["k_top"]), int(p["cbow_bits"]))
    M("features_run", str(feat_dir.relative_to(ROOT)))
    M("tokens_npz_sha256", hashing.sha256_file(feat_dir / "tokens.npz"))
    out = run_dir()
    cells_path = out / "cells.jsonl"
    done = set()
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], str(r["budget"]), r["seed"], r["classifier"], r["features"]))
    plan = p["plan"]
    if args.plan_items:
        plan = [plan[int(i)] for i in args.plan_items.split(",")]
        M("plan_items", args.plan_items)
    n_cells = 0
    t_start = time.time()
    for item in plan:
        task, budgets, clf = item["task"], item["budgets"], item["classifier"]
        pool_rows, dev_rows = store.task_rows(task)
        y_pool, y_dev = store.y(task)
        n_classes = TASKS[task]["n_classes"]
        for budget in budgets:
            for seed in item.get("seeds", p["seeds"]):
                lab = TR.label_sample(pool_rows.shape[0], budget, TASKS[task]["code"], seed)
                yl = y_pool[lab]
                for fs in item["features"]:
                    key = (task, str(budget), seed, clf, fs)
                    if key in done:
                        continue
                    past_deadline = bool(p.get("deadline_utc")) and time.time() > __import__("calendar").timegm(time.strptime(p["deadline_utc"], "%Y-%m-%dT%H:%M:%SZ"))
                    if (p.get("wall_cap_s") and time.time() - t_start > p["wall_cap_s"]) or past_deadline:
                        log(f"wall cap {p.get('wall_cap_s')} s or deadline {p.get('deadline_utc')} reached: stop before {key}")
                        M("stopped_by_wall_cap", True, before_cell=list(map(str, key)))
                        flush()
                        return 1 if FAILED else 0
                    t0 = time.time()
                    Xl, Xd, info = store.matrices(fs, task, pool_rows[lab], dev_rows, pool_rows, for_tm=(clf == "tm"))
                    t_feat = time.time() - t0
                    n_lab = int(lab.shape[0])
                    if clf == "lr":
                        folds = int(p["lr_folds_small"]) if n_lab <= int(p["folds_small_max_n"]) else int(p["lr_folds_large"])
                        c_grid = p["c_grid_full"] if budget == "full" else p["c_grid"]
                        r = TR.lr_cell(Xl, yl, Xd, y_dev, seed, dense=fs.startswith("cbow"), max_iter=int(p["lr_max_iter"]),
                                       c_grid=c_grid, max_folds=folds)
                    else:
                        grid = [dict(g, epochs=item.get("tm_epochs", p["tm_epochs"])) for g in p["tm_grid"]]
                        r = TR.tm_cell(Xl, yl, Xd, y_dev, seed, grid, n_classes, max_folds=int(p["tm_folds"]))
                    rec = {"task": task, "budget": budget, "seed": seed, "classifier": clf, "features": fs,
                           "n_labelled": int(lab.shape[0]), "n_dev": int(dev_rows.shape[0]), "accuracy": r.accuracy,
                           "choice": r.choice, "cv": r.cv, "fit_s": r.fit_s, "feature_s": t_feat, "dim": info["dim"],
                           "labelled_classes": int(np.unique(yl).shape[0]),
                           "pred_sha256": hashing.sha256_bytes(np.asarray(r.pred, np.int64).astype("<i8").tobytes()),
                           "pred": "".join(str(int(v)) for v in r.pred)}
                    with open(cells_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    n_cells += 1
                    log(f"{task} n={budget} seed={seed} {clf} {fs}: acc {r.accuracy:.4f} (choice {r.choice}, "
                        f"{r.fit_s:.1f} s fit, {t_feat:.1f} s features, dim {info['dim']})")
        M("cells_written", n_cells)
        M("memory", mem_gib())
        flush()
    M("cells_written", n_cells)
    flush()
    return 1 if FAILED else 0


# ------------------------------------------------------------------ report
def stage_report(args) -> int:
    """Merge the cells.jsonl of the classify runs; t-intervals over seeds; paired differences; the EARLY VERDICT.

    config params.roles: {role_name: {"classifier": "lr" | "tm", "verdict": bool, "trained": fs, "trained_e1": fs,
    "young": fs, "bag": [fs, ...]}}. Roles with verdict true are the pre-declared read-outs of the verdict (fixed
    in the config before any classification output existed); the others are descriptive. 'bag' is the STRONGER of
    its feature sets in each cell (the smaller trained-minus-bag difference)."""
    cfg = json.loads(Path(args.config).read_text())
    p = cfg["params"]
    cells = []
    for rd in p["classify_runs"]:
        for line in (ROOT / rd / "cells.jsonl").read_text().splitlines():
            r = json.loads(line)
            r["run"] = rd
            cells.append(r)
    by, preds, gold = {}, {}, {}
    for r in cells:
        key = (r["task"], str(r["budget"]), r["classifier"], r["features"])
        if r["seed"] in by.get(key, {}):
            check(f"no_duplicate_cell.{'.'.join(map(str, key))}.{r['seed']}", False)
        by.setdefault(key, {})[r["seed"]] = r["accuracy"]
        preds.setdefault(key, {})[r["seed"]] = np.frombuffer(r["pred"].encode("ascii"), np.uint8) - 48
    feat_dir = ROOT / p["features_run"]
    z = np.load(feat_dir / "tokens.npz")
    labels = json.loads(str(z["labels"]))
    for t in labels:
        gold[t] = np.asarray(labels[t]["dev"], np.int64)
    # consistency: stored predictions reproduce the stored accuracies
    worst = 0.0
    for key, sp in preds.items():
        for s, pr in sp.items():
            worst = max(worst, abs(float(np.mean(pr == gold[key[0]])) - by[key][s]))
    check("stored_predictions_reproduce_accuracy", worst < 1e-12, worst=worst)
    # cells computed before the used-column restriction of transfer._fit_lr (same model): re-run a sample with
    # the current code and compare the prediction hashes
    recheck = p.get("recheck_cells", [])
    if recheck:
        store = FeatureStore(feat_dir, p["tasks"], int(cfg["params"].get("k_top", 2048)), int(cfg["params"].get("cbow_bits", 8)))
        ccfg = json.loads((ROOT / p["classify_config"]).read_text())["params"]
        for rc in recheck:
            t, b, seed, fs = rc["task"], rc["budget"], rc["seed"], rc["features"]
            old = [r for r in cells if r["task"] == t and str(r["budget"]) == str(b) and r["seed"] == seed and r["classifier"] == "lr" and r["features"] == fs]
            if not old:
                continue
            pool_rows, dev_rows = store.task_rows(t)
            y_pool, y_dev = store.y(t)
            lab = TR.label_sample(pool_rows.shape[0], b, TASKS[t]["code"], seed)
            Xl, Xd, _ = store.matrices(fs, t, pool_rows[lab], dev_rows, pool_rows, for_tm=False)
            folds = int(ccfg["lr_folds_small"]) if lab.shape[0] <= int(ccfg["folds_small_max_n"]) else int(ccfg["lr_folds_large"])
            r = TR.lr_cell(Xl, y_pool[lab], Xd, y_dev, seed, dense=fs.startswith("cbow"), max_iter=int(ccfg["lr_max_iter"]),
                           c_grid=ccfg["c_grid"], max_folds=folds)
            sha = hashing.sha256_bytes(np.asarray(r.pred, np.int64).astype("<i8").tobytes())
            check(f"recheck.{t}.{b}.{seed}.{fs}.same_predictions_with_used_column_fit", sha == old[0]["pred_sha256"],
                  old_accuracy=old[0]["accuracy"], new_accuracy=r.accuracy, old_choice=old[0]["choice"], new_choice=r.choice)
    table = {}
    for (t, b, c, fs), accs in sorted(by.items()):
        table[f"{t}.{b}.{c}.{fs}"] = {**TR.t_interval([accs[s] for s in sorted(accs)]), "seeds": sorted(accs)}
    M("cells", table, note="accuracy on the validation/dev split: mean and Student-t 95 percent interval over the label-sample seeds (the interval does not include dev-set sampling noise; see paired_dev_bootstrap)")
    roles = p["roles"]
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 624]))

    def dev_boot(t, b, clf, fa, fb, B=2000):
        """Paired bootstrap over dev examples of mean-over-seeds accuracy(fa) - accuracy(fb), common seeds."""
        A, Bp = preds.get((t, str(b), clf, fa)), preds.get((t, str(b), clf, fb))
        seeds = sorted(set(A) & set(Bp))
        g = gold[t]
        d = np.mean([(A[s] == g).astype(np.float64) - (Bp[s] == g).astype(np.float64) for s in seeds], axis=0)
        idx = rng.integers(0, d.shape[0], size=(B, d.shape[0]))
        st = d[idx].mean(axis=1)
        return {"mean": float(d.mean()), "lo": float(np.percentile(st, 2.5)), "hi": float(np.percentile(st, 97.5)), "n_dev": int(d.shape[0])}

    def diff_ci(t, b, role, a, ref):
        r = roles.get(role)
        if r is None or a not in r or ref not in r:
            return None
        clf = r["classifier"]
        fa = r[a]
        cand = r[ref] if isinstance(r[ref], list) else [r[ref]]
        best = None
        for fb in cand:
            A, Bv = by.get((t, str(b), clf, fa)), by.get((t, str(b), clf, fb))
            if not A or not Bv:
                continue
            seeds = sorted(set(A) & set(Bv))
            if len(seeds) < 2:
                continue
            ci = TR.t_interval([A[s] - Bv[s] for s in seeds])
            ci["a_features"], ci["ref_features"] = fa, fb
            ci["dev_bootstrap"] = dev_boot(t, b, clf, fa, fb)
            if best is None or ci["mean"] < best["mean"]:
                best = ci
        return best

    budgets_seen = sorted({(t, b) for (t, b, _, _) in by}, key=lambda x: (x[0], (int(x[1]) if x[1].isdigit() else 10 ** 9)))
    verdict_roles = [k for k, v in roles.items() if v.get("verdict")]
    all_cells = [(t, int(b) if b.isdigit() else b, role) for (t, b) in budgets_seen for role in verdict_roles]
    v = TR.verdict(diff_ci, verdict_roles, tasks=tuple(p["verdict_tasks"]), budgets=tuple(p["verdict_budgets"]),
                   margin=float(p["verdict_margin"]), all_cells=all_cells)
    missing = [f"{t}.{b}.{role}" for t in p["verdict_tasks"] for b in p["verdict_budgets"] for role in verdict_roles
               if diff_ci(t, b, role, "trained", "young") is None or diff_ci(t, b, role, "trained", "bag") is None]
    v["missing_verdict_cells"] = missing
    if missing:
        v["verdict_note"] = "some verdict cells are missing: the verdict is computed on the cells present"
    M("early_verdict", v, label="derived", rule=p["verdict_rule"])
    diffs = {}
    for (t, b) in budgets_seen:
        for role in roles:
            for a, ref in (("trained", "young"), ("trained", "bag"), ("trained", "trained_e1"), ("trained_e1", "young"), ("young", "bag")):
                ci = diff_ci(t, b, role, a, ref)
                if ci is not None:
                    diffs[f"{t}.{b}.{role}.{a}_minus_{ref}"] = ci
    M("paired_differences", diffs, label="derived",
      note="per-seed paired differences of accuracy (Student-t 95 percent interval over seeds) plus a paired bootstrap over dev examples of the mean-over-seeds difference (2000 resamples); 'bag' = the stronger bag-of-tokens variant of the cell")
    lines = []
    for (t, b, c, fs), accs in sorted(by.items()):
        ci = table[f"{t}.{b}.{c}.{fs}"]
        lines.append(f"{t:7s} {b:>5s} {c:3s} {fs:18s} {ci['mean'] * 100:6.2f}" + (f" [{ci['lo'] * 100:6.2f}, {ci['hi'] * 100:6.2f}]" if ci["lo"] is not None else "") + f" n_seeds={ci['n']}")
    (run_dir() / "table.txt").write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    for k, ci in diffs.items():
        if "trained_minus_young" in k or "trained_minus_bag" in k:
            log(f"{k}: {ci['mean'] * 100:+.2f} [{ci['lo'] * 100:+.2f}, {ci['hi'] * 100:+.2f}] vs {ci['ref_features']}; dev bootstrap [{ci['dev_bootstrap']['lo'] * 100:+.2f}, {ci['dev_bootstrap']['hi'] * 100:+.2f}]")
    log(f"EARLY VERDICT: {v['verdict']}; trained beats young (interval above 0) in {len(v['trained_beats_young_cells'])} verdict-role cells; missing {missing}")
    flush()
    return 1 if FAILED else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "features", "classify", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks", default="sst2,agnews,trec")
    ap.add_argument("--banks", default="cp_e4,cp_e1,young")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--features-run", default=None, help="classify: run dir (relative to the project) holding tokens.npz and features_<bank>.npz")
    ap.add_argument("--plan-items", default=None, help="classify: comma-separated indices into the config's plan")
    ap.add_argument("--peak-gib", type=float, default=10.0, help="features stage: host-memory peak per bank used by the headroom wait")
    args = ap.parse_args()
    M("args", vars(args))
    M("status", "EXPLORATORY: not pre-registered, no docs/13 threshold tested; downstream train and validation/dev only", label="unverified")
    np.random.seed(20260924)
    rc = {"data": stage_data, "features": stage_features, "classify": stage_classify, "report": stage_report}[args.stage](args)
    M("failed_checks", FAILED)
    flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
