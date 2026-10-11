"""R1 PRETEXT TOURNAMENT of docs/13 amendment A7 (family H17): train an arm's clause bank self-supervised on
WikiText-103, freeze it, and judge it as a text representation with few labels against the learning-credit
controls. Seed 100 is EXPLORATORY (A7: "R0 and every seed-100 run are exploratory"); every cell here uses
only the train splits of the downstream tasks (the exploration dev carve), never a test split and never
SST-2 validation (src/tmlm/ssl/downstream.py refuses them). The WikiText-103 test split is never loaded.

Arms (src/tmlm/ssl/pretext.py): A0 default masked, A1 masked + F1, A2 F1 + F3 + F4, A3 F2, A4 causal,
A5 diffusion, A6 A1 with a random kind_map. Common: V 8000, C 65,536, the first 20M training tokens cut
at a document boundary, 4 epochs, the docs/11 CUDA engine (bit-exact with the CPU engine).

Stages (each its own logged run; config experiments/cvtm/configs/ssl_r1_pretext.json):
  data       (CPU) write once, or re-verify, docs/notes/preregistration-data-hashes-A7-<task>.json for
             agnews, sst2, trec (train files only; carve hashes).
  train      (GPU) train the arm; save checkpoint_c0y.npz (C0y: after ceil(1 percent) of the batches of
             epoch 1) and checkpoint_epoch<k>.npz after every epoch.
  features   (GPU) the downstream document set (per task: the dev carve plus the union of the labelled
             samples of every budget and draw; `--full` adds the whole pool), BPE-8k with the WikiText-103
             tokenizer (src/tmlm/ssl/features.py); fired-clause codes of the final bank and of C0y over every
             position of every document (masked and diffusion: nothing hidden except the position itself;
             causal: the left context), per-clause fire counts over the pretraining prefix (the 8,192
             most-firing clauses = the dimension-matched columns), and G+ (src/tmlm/ssl/controls.py:
             selection on the pretraining prefix, feature map on the documents).
  baselines  (CPU) PPMI-SVD token vectors over the pretraining prefix (window 8, 300 dims, p 0.5).
  probe      (CPU) L2 logistic regression (4-fold CV inside the labelled sample) and the Coalesced TM probe
             on the dev carve, per task, budget and draw, for the named feature sets; cells.jsonl.
             Arm sets: bank.all, bank.dim, c0y.all, c0y.dim, gplus.all, gplus.dim. Baseline sets (arm
             independent, run once): bow, cluster, ppmi.
  report     (CPU) Q_text per feature set, budget and probe over draws; paired differences (per draw) of the
             bank against C0y, G+, the stronger of the two, and bag of words; the tournament table.
  all        SMOKE ONLY: train, features, baselines, probe and report in one process.

Launch commands (detached; CPU slot FIRST, then the GPU lock, docs/12 F25; the memory guard in group scope):
    cd studies/tm_language_model
    # 0. data hashes, once (writes docs/notes/preregistration-data-hashes-A7-{agnews,sst2,trec}.json)
    setsid nohup scripts/cpu_slot.sh 1 -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 8 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_data --config experiments/cvtm/configs/ssl_r1_pretext.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage data --config experiments/cvtm/configs/ssl_r1_pretext.json \\
        > /dev/null 2>&1 &
    # 1. per arm A in A0..A6 (seed 100 exploratory; later seeds 1 to 3 for the top two arms)
    setsid nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 16 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_train_A1_seed100_exploratory --config experiments/cvtm/configs/ssl_r1_pretext.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage train --arm A1 --seed 100 --config experiments/cvtm/configs/ssl_r1_pretext.json \\
        > /dev/null 2>&1 &
    # 2. after the train run: features (needs the train run dir)
    setsid nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 24 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_features_A1_seed100_exploratory --config experiments/cvtm/configs/ssl_r1_pretext.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage features --arm A1 --seed 100 --train-run results/<ts>_ssl_r1_train_A1_seed100_exploratory \\
        --config experiments/cvtm/configs/ssl_r1_pretext.json > /dev/null 2>&1 &
    # 3. baselines, once (any features run supplies the identical document set)
    setsid nohup scripts/cpu_slot.sh 1 -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 16 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_baselines --config experiments/cvtm/configs/ssl_r1_pretext.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage baselines --features-run results/<ts>_ssl_r1_features_A1_seed100_exploratory \\
        --config experiments/cvtm/configs/ssl_r1_pretext.json > /dev/null 2>&1 &
    # 4. probes: per arm (arm sets), and once for the baseline sets
    setsid nohup scripts/cpu_slot.sh 1 -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 24 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_probe_A1_seed100_exploratory --config experiments/cvtm/configs/ssl_r1_pretext.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage probe --features-run results/<ts>_ssl_r1_features_A1_seed100_exploratory \\
        --sets bank.all,bank.dim,c0y.all,c0y.dim,gplus.all,gplus.dim --config experiments/cvtm/configs/ssl_r1_pretext.json > /dev/null 2>&1 &
    #    baseline sets: --stage probe --features-run <any features run> --baselines-run results/<ts>_ssl_r1_baselines --sets bow,cluster,ppmi --name ssl_r1_probe_baselines
    # 5. report: --stage report --probe-runs <comma separated probe run dirs>  (name ssl_r1_report_seed100_exploratory)
  The smoke (logged, labelled smoke):
    setsid nohup scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group --limit-gib 12 -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_r1_pretext_smoke --config experiments/cvtm/configs/ssl_r1_pretext_smoke.json -- \\
        .venv-tm/bin/python -u experiments/ssl_r1_pretext.py --stage all --arm A1 --seed 100 --smoke --config experiments/cvtm/configs/ssl_r1_pretext_smoke.json \\
        > /dev/null 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.ssl import controls as K  # noqa: E402
from tmlm.ssl import downstream as D  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import probes as PR  # noqa: E402

T0 = time.time()
CORPUS = "wikitext103"
ARM_SETS = ("bank.all", "bank.dim", "c0y.all", "c0y.dim", "gplus.all", "gplus.dim")
BASELINE_SETS = ("bow", "cluster", "ppmi")


# ------------------------------------------------------------------ bookkeeping
class Rec:
    """Labelled metrics; every entry carries exploratory (and smoke) flags."""

    def __init__(self, smoke: bool) -> None:
        self.smoke = smoke
        self.metrics: dict = {}
        self.failed: list[str] = []

    def log(self, msg: str) -> None:
        print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        e = {"value": value, "label": label, "exploratory": True, **extra}
        if self.smoke:
            e["smoke"] = True
        self.metrics[name] = e

    def check(self, name: str, ok: bool, **extra) -> None:
        self.m(f"check.{name}", bool(ok), **extra)
        if not ok:
            self.failed.append(name)
            self.log(f"CHECK FAILED: {name} {extra}")

    def flush(self) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            runlog.write_metrics(self.metrics, on_conflict="replace")


_SCRATCH: list[Path] = []


def out_dir(stage: str) -> Path:
    """The run directory of scripts/run_logged.py; without it (tests), one scratch directory per process."""
    d = runlog.current_run_dir()
    if d is not None:
        return Path(d)
    if not _SCRATCH:
        s = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_r1_{stage}_{os.getpid()}_{int(time.time())}"
        s.mkdir(parents=True, exist_ok=True)
        _SCRATCH.append(s)
    return _SCRATCH[0]


def rp(path: str | None) -> Path | None:
    """A run directory argument, relative to the project root unless absolute."""
    if path is None:
        return None
    q = Path(path.strip())
    return q if q.is_absolute() else ROOT / q


def mem_info() -> dict:
    out = {"rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20}
    try:
        out["cgroup_current_gib"] = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2 ** 30
    except OSError:
        pass
    return out


def load_params(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["params"]


def dataset_hashes(d: dict) -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes(d, on_conflict="replace")


# ------------------------------------------------------------------ corpus and engines
def load_prefix(p: dict, rec: Rec):
    from tmlm.data.corpus_store import load_split, split_dir

    full = load_split(CORPUS, int(p["vocab"]), "train")
    prefix = P.slice_prefix(full, int(p["n_tokens"]))
    meta = json.loads((split_dir(CORPUS, int(p["vocab"]), "train") / "meta.json").read_text())
    dataset_hashes({f"{CORPUS}_bpe{p['vocab']}_train_tok": meta["sha256"]["tok.npy"],
                    f"{CORPUS}_bpe{p['vocab']}_train_doc_start": meta["sha256"]["doc_start.npy"],
                    f"{CORPUS}_bpe{p['vocab']}_train_kind_map": meta["sha256"]["kind_map.npy"],
                    f"{CORPUS}_bpe{p['vocab']}_train_sym_freq": meta["sha256"]["sym_freq.npy"]})
    rec.m("prefix.tokens", int(prefix.tok.shape[0]), note=f"first {p['n_tokens']} tokens cut at a document boundary")
    rec.m("prefix.docs", int(prefix.doc_start.shape[0] - 1))
    rec.m("prefix.tok_sha256", hashing.sha256_bytes(np.ascontiguousarray(prefix.tok).astype("<u2").tobytes()))
    return prefix


def make_engine(kind: str, cfg, corpus):
    if kind == "cuda":
        from tmlm.engine.cuda.cuda_engine import CudaEngine
        return CudaEngine(cfg, corpus=corpus)
    if kind == "cpu":
        from tmlm.engine.cpu.cpu_engine import CpuEngine
        return CpuEngine(cfg, corpus=corpus)
    if kind == "oracle":
        from tmlm.engine.oracle.oracle import OracleEngine
        return OracleEngine(cfg, corpus=corpus)
    raise ValueError(kind)


def train_arm(ad: P.ArmData, cfg, epochs: int, out: Path, engine_kind: str, rec: Rec, young_fraction: float,
              yield_gpu: bool = True) -> dict:
    """Train `epochs` epochs over the arm's example sets; C0y checkpoint after ceil(young_fraction x batches of
    epoch 0) batches; a checkpoint after every epoch. Returns {'c0y': hash, 'epochs': {k: hash}}."""
    from tmlm.engine.checkpoint import save_checkpoint

    hashes: dict[str, Any] = {"epochs": {}}
    eng = None
    state = None
    for e in range(int(epochs)):
        if eng is None or ad.targets_vary_by_epoch:
            corpus = ad.corpus(e)
            t_b = time.time()
            if eng is not None:
                state = eng.state()
                del eng
            eng = make_engine(engine_kind, cfg, corpus)
            if state is not None:
                eng.load_state(state)
            rec.m(f"train.epoch{e + 1}.engine_build_s", time.time() - t_b)
        n_ex = eng.n_examples
        nb = eng.n_batches()
        rec.m(f"train.epoch{e + 1}.targets", int(n_ex))
        young_b = K.c0y_batches(n_ex, cfg.batch, young_fraction) if e == 0 else -1
        diag: dict[str, int] = {}
        t0 = time.time()
        for b in range(nb):
            d = eng.train_batch(e, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + int(v)
            if b + 1 == young_b:
                h = save_checkpoint(out / "checkpoint_c0y.npz", eng, epoch=0, batch_index=young_b, compress=True,
                                    extra={"what": "C0y: the same run stopped at 1 percent of epoch 1 (A7)",
                                           "batches": young_b, "batches_in_epoch": nb, "examples": int(min(n_ex, young_b * cfg.batch))})
                hashes["c0y"] = h
                rec.m("c0y.batches", int(young_b), note=f"ceil({young_fraction} x {nb} batches of epoch 1)")
                rec.m("c0y.examples", int(min(n_ex, young_b * cfg.batch)))
                rec.m("c0y.state_hash", h)
                rec.m("c0y.alive", int(eng.state()["alive"].sum()))
        if hasattr(eng, "synchronize"):
            eng.synchronize()
        wall = time.time() - t0
        st = eng.state()
        h = save_checkpoint(out / f"checkpoint_epoch{e + 1}.npz", eng, epoch=e + 1, batch_index=0, compress=True)
        hashes["epochs"][e + 1] = h
        pre = f"train.epoch{e + 1}"
        rec.m(f"{pre}.wall_s", wall, note="training only")
        rec.m(f"{pre}.examples_per_s", diag.get("n_examples", 0) / max(1e-9, wall))
        rec.m(f"{pre}.n_examples", int(diag.get("n_examples", 0)))
        for k in ("n_candidates", "n_fired", "n_events_applied", "n_events_dropped_cap", "n_tgt_gate_pass",
                  "n_neg_gate_pass", "n_inserted", "n_neg_inserted"):
            rec.m(f"{pre}.{k}_per_example", diag.get(k, 0) / max(1, diag.get("n_examples", 0)))
        for k in ("n_seeded", "n_recycled", "n_events_dropped_cap"):
            rec.m(f"{pre}.{k}", int(diag.get(k, 0)))
        rec.m(f"{pre}.alive", int(st["alive"].sum()))
        rec.m(f"{pre}.state_hash", h)
        if hasattr(eng, "device_memory_used"):
            rec.m(f"{pre}.device_pool_bytes", int(eng.device_memory_used()), unit="bytes")
        rec.m(f"{pre}.memory", mem_info())
        rec.log(f"epoch {e + 1}: {diag.get('n_examples', 0)} examples in {wall:.0f} s, alive {int(st['alive'].sum())}, hash {h[:12]}")
        rec.flush()
        if yield_gpu and engine_kind == "cuda":
            from tmlm.common.locks import yield_gpu_if_requested
            if yield_gpu_if_requested():
                rec.log("yielded the GPU to an exclusive job")
    return hashes


def stage_train(args, p: dict, rec: Rec) -> None:
    out = out_dir("train")
    arm = P.ARMS[args.arm]
    prefix = load_prefix(p, rec)
    ad = P.arm_data(arm, prefix, int(p["vocab"]), float(p["f1_t"]))
    cfg = P.engine_config(arm, args.seed, int(p["vocab"]), int(p["n_clauses"]))
    rec.m("arm", arm.to_dict())
    rec.m("engine_config", cfg.to_dict())
    rec.m("arm_data", ad.info)
    if arm.f1:
        rec.m("f1.expected_fraction", P.expected_f1_fraction(P.token_counts(prefix.tok, int(p["vocab"])), float(p["f1_t"])), label="derived")
    if arm.random_kind_map:
        np.save(out / "kind_map_random.npy", ad.kind_map)
        rec.m("random_kind_map.sha256", hashing.sha256_bytes(ad.kind_map.astype("<u2").tobytes()))
    rec.flush()
    hashes = train_arm(ad, cfg, int(p["epochs"]), out, args.engine, rec, float(p["young_fraction"]))
    rec.m("final.state_hash", hashes["epochs"][int(p["epochs"])])
    rec.m("train.wall_total_s", time.time() - T0)
    rec.flush()


# ------------------------------------------------------------------ downstream document set
def doc_set(p: dict, rec: Rec, full: bool) -> tuple[F.TokenDocs, dict, dict]:
    """Every task's dev carve plus the union of its labelled samples (and the pool with --full), tokenized.
    layout[task] = {'dev': [a, b), 'lab': [c, d), 'lab_pool_rows': pool rows of the lab docs (ascending)}."""
    from tmlm.data import bpe

    tok = bpe.load_tokenizer(int(p["vocab"]))
    parts, layout, labels = [], {}, {}
    base = 0
    for task in p["tasks"]:
        td = D.load_task(task, allow_download=False, verify=True)
        code = D.TASKS[task]["code"]
        budgets = list(p["budgets"]) + (["full"] if full else [])
        union = PR.sample_union(td.pool_y, budgets, p["draws"], code)
        dev_docs = F.encode(tok, td.dev_text)
        lab_docs = F.encode(tok, [td.pool_text[i] for i in union])
        parts += [dev_docs, lab_docs]
        layout[task] = {"dev": [base, base + dev_docs.n_doc], "lab": [base + dev_docs.n_doc, base + dev_docs.n_doc + lab_docs.n_doc],
                        "lab_pool_rows": union.tolist()}
        base += dev_docs.n_doc + lab_docs.n_doc
        labels[task] = {"pool_y": td.pool_y.tolist(), "dev_y": td.dev_y.tolist()}
        rec.m(f"data.{task}.n_train", td.n_train)
        rec.m(f"data.{task}.pool", int(td.pool_y.shape[0]))
        rec.m(f"data.{task}.dev", int(td.dev_y.shape[0]))
        rec.m(f"data.{task}.confirm_excluded", td.n_confirm_excluded)
        rec.m(f"data.{task}.labelled_union_docs", int(union.shape[0]))
        rec.m(f"data.{task}.train_sha256", td.train_sha256)
        dataset_hashes({f"{task}_train": td.train_sha256})
    docs, _ = F.concat(parts)
    rec.m("docs.n_docs", docs.n_doc)
    rec.m("docs.n_tokens", docs.n_tok, note="text tokens plus one END token per document")
    rec.m("docs.tok_sha256", hashing.sha256_bytes(docs.tok.astype("<u2").tobytes() + docs.doc_start.astype("<u4").tobytes()))
    return docs, layout, labels


# ------------------------------------------------------------------ bank featurisation
def hidden_for(cfg, n_tok: int) -> np.ndarray | None:
    """Masked and diffusion banks: a zero bitmap (nothing hidden but the position itself); causal: None."""
    return np.zeros(n_tok, np.uint8) if cfg.mode in ("masked", "diffusion") else None


def bank_codes(eng, cfg, docs: F.TokenDocs, chunk: int, rec: Rec, name: str) -> F.SparseDocs:
    """Per document: number of positions where each clause fires (all positions of the document)."""
    hidden = hidden_for(cfg, docs.n_tok)
    if not hasattr(eng, "_forward"):  # CPU / oracle engines (tests): public API, per position
        ev_d, ev_j = [], []
        dop = docs.doc_of_pos()
        for q in range(docs.n_tok):
            a = eng.fired_clauses(0, q, hidden)
            ev_d.append(np.full(a.shape[0], dop[q], np.int64))
            ev_j.append(a.astype(np.int64))
        return F.aggregate_doc_counts(np.concatenate(ev_d), np.concatenate(ev_j), docs.n_doc, cfg.n_clauses)
    cp = eng.cp
    bitmap = eng._bitmap(hidden)
    ds = docs.doc_start.astype(np.int64)
    dop_dev = cp.asarray(docs.doc_of_pos().astype(np.int32))
    parts, fired_total = [], 0
    t0 = time.time()
    for d0, d1 in K.doc_chunks(ds, chunk):
        lo, hi = int(ds[d0]), int(ds[d1])
        eng._sync_state_struct()
        fw = eng._forward(0, cp.arange(lo, hi, dtype=cp.uint32), bitmap, 0)
        eng._marks = []
        fe, fj = fw.fired_ex, fw.fired_j
        n_f = int(fe.size)
        fired_total += n_f
        if n_f:
            drel = (dop_dev[fe.astype(cp.int64) + lo] - d0).astype(cp.uint64)
            key = (drel << cp.uint64(32)) | fj.astype(cp.uint64)
            u, c = cp.unique(key, return_counts=True)
            uh, ch = u.get(), c.get()
            dd = (uh >> np.uint64(32)).astype(np.int64)
            ids = (uh & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            cnt = ch.astype(np.uint32)
        else:
            dd, ids, cnt = np.zeros(0, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint32)
        ip = np.zeros(d1 - d0 + 1, np.int64)
        ip[1:] = np.cumsum(np.bincount(dd, minlength=d1 - d0))
        parts.append(F.SparseDocs(ip, ids, cnt, int(cfg.n_clauses)))
        del fw
    eng.synchronize()
    wall = time.time() - t0
    rec.m(f"features.{name}.wall_s", wall)
    rec.m(f"features.{name}.positions_per_s", docs.n_tok / max(1e-9, wall))
    rec.m(f"features.{name}.fired_per_position", fired_total / max(1, docs.n_tok))
    cp.get_default_memory_pool().free_all_blocks()
    return F.merge_rows(parts)


def fire_counts(eng, cfg, n_tok: int, chunk: int) -> np.ndarray:
    """u64 [C]: number of pretraining-prefix positions where each clause fires (nothing hidden but the position)."""
    C = int(cfg.n_clauses)
    hidden = hidden_for(cfg, n_tok)
    if not hasattr(eng, "_forward"):
        out = np.zeros(C, np.uint64)
        for q in range(n_tok):
            a = eng.fired_clauses(0, q, hidden)
            out[a.astype(np.int64)] += np.uint64(1)
        return out
    cp = eng.cp
    bitmap = eng._bitmap(hidden)
    acc = cp.zeros(C, dtype=cp.uint64)
    for lo in range(0, n_tok, chunk):
        hi = min(n_tok, lo + chunk)
        eng._sync_state_struct()
        fw = eng._forward(0, cp.arange(lo, hi, dtype=cp.uint32), bitmap, 0)
        eng._marks = []
        if fw.fired_j.size:
            acc += cp.bincount(fw.fired_j.astype(cp.int64), minlength=C).astype(cp.uint64)
        del fw
    eng.synchronize()
    return acc.get()


def pipeline_checks(eng, cfg, docs: F.TokenDocs, sd: F.SparseDocs, rec: Rec, name: str, n_docs: int = 12) -> None:
    """Our per-document aggregate equals the CPU reference over the public fired_clauses on sample documents;
    two positions equal fired_clauses_bruteforce (CUDA engine)."""
    hidden = hidden_for(cfg, docs.n_tok)
    rs = np.random.default_rng(12345)
    chk = np.sort(rs.choice(docs.n_doc, size=min(n_docs, docs.n_doc), replace=False))
    ok_agg, ok_bf = True, True
    for i, d in enumerate(chk.tolist()):
        lo, hi = int(docs.doc_start[d]), int(docs.doc_start[d + 1])
        ev_j = []
        for q in range(lo, hi):
            a = eng.fired_clauses(0, q, hidden)
            if i < 2 and q == lo + (hi - lo) // 2 and hasattr(eng, "fired_clauses_bruteforce"):
                ok_bf &= np.array_equal(np.sort(eng.fired_clauses_bruteforce(0, q, hidden)), np.sort(a))
            ev_j.append(a.astype(np.int64))
        j = np.concatenate(ev_j)
        ref = F.aggregate_doc_counts(np.zeros(j.shape[0], np.int64), j, 1, sd.n_cols)
        got = sd.rows(np.asarray([d]))
        ok_agg &= np.array_equal(ref.ids, got.ids) and np.array_equal(ref.cnt, got.cnt)
    rec.check(f"features.{name}.doc_aggregate_equals_public_fired_clauses", ok_agg, docs=int(chk.shape[0]))
    rec.check(f"features.{name}.fired_equal_bruteforce", ok_bf, positions=2)


def save_sparse(path: Path, sd: F.SparseDocs, **extra) -> str:
    np.savez(path, indptr=sd.indptr, ids=sd.ids, cnt=sd.cnt, n_cols=np.int64(sd.n_cols), **extra)
    return hashing.sha256_file(path)


def load_sparse(path: Path) -> F.SparseDocs:
    z = np.load(path)
    return F.SparseDocs(z["indptr"], z["ids"], z["cnt"], int(z["n_cols"]))


def read_metric(run_dir: Path, name: str, rec: Rec | None = None):
    """A metric of an earlier run (its metrics.json), or of this process when run_dir is this run."""
    f = run_dir / "metrics.json"
    if f.exists():
        m = json.loads(f.read_text())
        if name in m:
            return m[name]["value"]
    if rec is not None and name in rec.metrics:
        return rec.metrics[name]["value"]
    raise KeyError(f"{name} not found in {f}")


def stage_features(args, p: dict, rec: Rec, train_run: Path) -> Path:
    from tmlm.engine.checkpoint import load_checkpoint

    out = out_dir("features")
    arm = P.ARMS[args.arm]
    docs, layout, labels = doc_set(p, rec, args.full)
    np.savez(out / "tokens.npz", tok=docs.tok, doc_start=docs.doc_start, end_id=np.int64(docs.end_id),
             layout=json.dumps(layout), labels=json.dumps(labels), arm=args.arm, seed=np.int64(args.seed))
    rec.flush()
    prefix = load_prefix(p, rec)
    ad = P.arm_data(arm, prefix, int(p["vocab"]), float(p["f1_t"]))
    ref = ad.feature_tables()
    corpus_docs = F.corpus_arrays(docs, ref)
    corpus_prefix = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    chunk = int(p["forward_chunk"])
    for bank, fname in (("bank", f"checkpoint_epoch{int(p['epochs'])}.npz"), ("c0y", "checkpoint_c0y.npz")):
        cfg, st, meta = load_checkpoint(train_run / fname)
        logged = read_metric(train_run, "final.state_hash" if bank == "bank" else "c0y.state_hash", rec)
        rec.check(f"{bank}.state_hash_equals_train_run", meta["state_hash"] == logged, state_hash=meta["state_hash"])
        if cfg.mode != arm.mode or int(cfg.seed) != int(args.seed) or int(cfg.vocab) != int(p["vocab"]):
            raise SystemExit(f"{fname}: config does not match arm {arm.name} seed {args.seed}")
        rec.m(f"{bank}.alive", int(np.asarray(st["alive"]).sum()))
        t0 = time.time()
        eng = make_engine(args.engine, cfg, corpus_docs)
        eng.load_state(st)
        sd = bank_codes(eng, cfg, docs, chunk, rec, bank)
        pipeline_checks(eng, cfg, docs, sd, rec, bank)
        rec.m(f"features.{bank}.nnz", int(sd.ids.shape[0]))
        rec.m(f"features.{bank}.distinct_clauses_per_doc_mean", float(np.diff(sd.indptr).mean()))
        rec.m(f"features.{bank}.file_sha256", save_sparse(out / f"features_{bank}.npz", sd, state_hash=meta["state_hash"]))
        del eng
        eng = make_engine(args.engine, cfg, corpus_prefix)
        eng.load_state(st)
        t1 = time.time()
        fc = fire_counts(eng, cfg, int(prefix.tok.shape[0]), chunk)
        rec.m(f"prefix_fires.{bank}.wall_s", time.time() - t1)
        rec.m(f"prefix_fires.{bank}.clauses_firing", int((fc > 0).sum()))
        rec.m(f"prefix_fires.{bank}.fired_per_position", float(fc.sum()) / max(1, prefix.tok.shape[0]))
        dim = K.top_columns_by(fc.astype(np.int64), int(p["dim_matched"]))
        np.save(out / f"fires_{bank}.npy", fc)
        np.save(out / f"dim_{bank}.npy", dim)
        rec.m(f"features.{bank}.dim_matched_columns", int(dim.shape[0]))
        del eng
        rec.log(f"{bank}: features and prefix fire counts in {time.time() - t0:.0f} s")
        if args.engine == "cuda":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        rec.m(f"memory.after_{bank}", mem_info())
        rec.flush()
    # G+
    xp = np
    if args.engine == "cuda":
        import cupy as xp  # noqa: F811
    t0 = time.time()
    g = K.gplus_select(xp, prefix.tok, prefix.doc_start, ad.kind_map, arm.mode, int(p["vocab"]), int(p["n_clauses"]),
                       chunk=int(p["gplus_chunk_positions"]), log=rec.log)
    g.save(out / "gplus_select.npz")
    rec.m("gplus.select", g.info)
    rec.m("gplus.select_wall_s", time.time() - t0)
    rec.m("gplus.examples", [g.describe(f) for f in range(min(8, g.n))], note="the most frequent features")
    t0 = time.time()
    sdg = K.gplus_features(xp, g, docs.tok, docs.doc_start, ad.kind_map, chunk=int(p["gplus_feature_chunk"]))
    rec.m("gplus.features_wall_s", time.time() - t0)
    # check: a few documents against the brute-force reference (pure Python)
    rs = np.random.default_rng(777)
    ok = True
    for d in np.sort(rs.choice(docs.n_doc, size=min(3, docs.n_doc), replace=False)).tolist():
        pos = range(int(docs.doc_start[d]), int(docs.doc_start[d + 1]))
        bf = K.gplus_bruteforce(docs.tok, docs.doc_start, ad.kind_map, arm.mode, int(p["vocab"]), positions=pos)
        key_to_f = {(int(a), int(b)): f for f, (a, b) in enumerate(zip(g.lit1, g.lit2))}
        exp = {key_to_f[k]: v for k, v in bf.items() if k in key_to_f}
        row = sdg.rows(np.asarray([d]))
        ok &= dict(zip(row.ids.tolist(), row.cnt.tolist())) == exp
    rec.check("gplus.features_equal_bruteforce", ok, docs=3)
    rec.m("features.gplus.nnz", int(sdg.ids.shape[0]))
    rec.m("features.gplus.file_sha256", save_sparse(out / "features_gplus.npz", sdg))
    np.save(out / "dim_gplus.npy", np.arange(min(int(p["dim_matched"]), g.n), dtype=np.int64))
    rec.m("memory.after_gplus", mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ baselines
def stage_baselines(args, p: dict, rec: Rec, features_run: Path) -> Path:
    out = out_dir("baselines")
    prefix = load_prefix(p, rec)
    t0 = time.time()
    E, info = K.ppmi_svd(np.asarray(prefix.tok), np.asarray(prefix.doc_start), int(p["vocab"]), int(p["ppmi"]["window"]),
                         int(p["ppmi"]["dim"]), float(p["ppmi"]["power"]))
    np.save(out / "ppmi_svd.npy", E)
    rec.m("ppmi_svd.info", info)
    rec.m("ppmi_svd.wall_s", time.time() - t0)
    rec.m("ppmi_svd.sha256", hashing.sha256_file(out / "ppmi_svd.npy"))
    rec.m("features_run", str(features_run))
    rec.m("memory", mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ probes
class Store:
    """Representations of one features run (plus the baselines run for ppmi)."""

    def __init__(self, features_run: Path, baselines_run: Path | None, p: dict) -> None:
        z = np.load(features_run / "tokens.npz")
        self.docs = F.TokenDocs(tok=z["tok"], doc_start=z["doc_start"], end_id=int(z["end_id"]))
        self.layout = json.loads(str(z["layout"]))
        self.labels = json.loads(str(z["labels"]))
        self.arm, self.seed = str(z["arm"]), int(z["seed"])
        self.fr, self.br, self.p = features_run, baselines_run, p
        self._cache: dict = {}

    def rows(self, task: str, pool_rows: np.ndarray) -> np.ndarray:
        lab = self.layout[task]
        union = np.asarray(lab["lab_pool_rows"], np.int64)
        k = np.searchsorted(union, pool_rows)
        if np.any(k >= union.shape[0]) or np.any(union[np.minimum(k, union.shape[0] - 1)] != pool_rows):
            raise KeyError("a labelled row is not in the featurised document set")
        return lab["lab"][0] + k

    def dev_rows(self, task: str) -> np.ndarray:
        a, b = self.layout[task]["dev"]
        return np.arange(a, b, dtype=np.int64)

    def sparse(self, name: str) -> F.SparseDocs:
        if name not in self._cache:
            if name in ("bank", "c0y", "gplus"):
                self._cache[name] = load_sparse(self.fr / f"features_{name}.npz")
            elif name == "bow":
                self._cache[name] = K.bag_of_words(self.docs, int(self.p["vocab"]))
            elif name == "cluster":
                from tmlm.data.corpus_store import load_split
                km = np.asarray(load_split(CORPUS, int(self.p["vocab"]), "train").kind_map)
                self._cache[name] = K.cluster_bag_of_words(self.docs, km)
            else:
                raise KeyError(name)
        return self._cache[name]

    def dim_cols(self, name: str) -> np.ndarray:
        return np.load(self.fr / f"dim_{name}.npy")

    def ppmi_docs(self) -> np.ndarray:
        if "ppmi" not in self._cache:
            if self.br is None:
                raise SystemExit("the ppmi set needs --baselines-run")
            self._cache["ppmi"] = K.ppmi_svd_docs(self.docs, np.load(self.br / "ppmi_svd.npy"))
        return self._cache["ppmi"]

    def lr_matrices(self, fs: str, lab: np.ndarray, dev: np.ndarray):
        """(Xl, Xd, dense)."""
        if fs == "ppmi":
            X = self.ppmi_docs()
            return X[lab], X[dev], True
        src, _, var = fs.partition(".")
        sd = self.sparse(src)
        cols = self.dim_cols(src) if var == "dim" else None
        return sd.rows(lab).to_csr("log1p", cols=cols), sd.rows(dev).to_csr("log1p", cols=cols), False

    def tm_matrices(self, fs: str, lab: np.ndarray, dev: np.ndarray):
        """Binary uint32 (Xl, Xd) for the TM probe, at most dim_matched columns (see the config)."""
        if fs == "ppmi":
            X = self.ppmi_docs()
            edges = F.quantile_edges(X[lab], int(self.p["tm_ppmi_bits"]))
            return F.thermometer(X[lab], edges), F.thermometer(X[dev], edges)
        src, _, var = fs.partition(".")
        sd = self.sparse(src)
        if src in ("bank", "c0y", "gplus"):
            if var != "dim":
                return None
            cols = self.dim_cols(src)
        else:
            cols = np.arange(sd.n_cols, dtype=np.int64)
        return K.binarise_counts(sd, lab, cols), K.binarise_counts(sd, dev, cols)


def stage_probe(args, p: dict, rec: Rec, features_run: Path, baselines_run: Path | None) -> Path:
    out = out_dir("probe")
    for src in (features_run, baselines_run):
        if src is not None and src != out and (src / "metrics.json").exists():
            failed = json.loads((src / "metrics.json").read_text()).get("failed_checks", {}).get("value")
            if failed:
                raise SystemExit(f"{src} has failed checks {failed}; not probing it")
    store = Store(features_run, baselines_run, p)
    sets = [s for s in args.sets.split(",") if s]
    clfs = [c for c in args.classifiers.split(",") if c]
    rec.m("probe.sets", sets)
    rec.m("probe.features_run", str(features_run))
    rec.m("probe.arm", store.arm)
    rec.m("probe.seed", store.seed)
    budgets = list(p["budgets"]) + (["full"] if args.full else [])
    cells = out / "cells.jsonl"
    n_cells = 0
    with open(cells, "a", encoding="utf-8") as fh:
        for task in p["tasks"]:
            code = D.TASKS[task]["code"]
            pool_y = np.asarray(store.labels[task]["pool_y"], np.int64)
            dev_y = np.asarray(store.labels[task]["dev_y"], np.int64)
            dev = store.dev_rows(task)
            for budget in budgets:
                for draw in (p["draws"] if budget != "full" else p["draws"][:1]):
                    pr = PR.label_sample(pool_y, budget, code, draw)
                    lab = store.rows(task, pr)
                    yl = pool_y[pr]
                    seed = PR.fold_seed(code, budget, draw)
                    for fs in sets:
                        for clf in clfs:
                            if clf == "tm" and budget not in p["tm_budgets"]:
                                continue
                            t0 = time.time()
                            if clf == "lr":
                                Xl, Xd, dense = store.lr_matrices(fs, lab, dev)
                                c = PR.lr_probe(Xl, yl, Xd, dev_y, seed, dense=dense, c_grid=p["c_grid"], folds=int(p["folds"]),
                                                max_iter=int(p["lr_max_iter"]), backend=p.get("lr_backend", "cpu"))
                            else:
                                mats = store.tm_matrices(fs, lab, dev)
                                if mats is None:
                                    continue
                                c = PR.tm_probe(mats[0], yl, mats[1], dev_y, seed, grid=p["tm_grid"], epochs=int(p["tm_epochs"]),
                                                folds=int(p["folds"]))
                            row = {"arm": store.arm, "seed": store.seed, "set": fs, "clf": clf, "task": task, "budget": budget,
                                   "draw": draw, "acc": c.accuracy, "choice": c.choice, "cv": c.cv, "n_labelled": c.n_labelled,
                                   "n_eval": c.n_eval, "class_counts": np.bincount(yl, minlength=D.TASKS[task]["n_classes"]).tolist(),
                                   "wall_s": time.time() - t0, "smoke": bool(args.smoke)}
                            fh.write(json.dumps(row, default=str) + "\n")
                            fh.flush()
                            n_cells += 1
                            rec.m(f"cell.{task}.{fs}.{clf}.{budget}.d{draw}", c.accuracy, n_labelled=c.n_labelled, n_eval=c.n_eval)
                    rec.log(f"{task} budget {budget} draw {draw}: {len(sets)} sets done")
                    rec.flush()
    rec.m("probe.cells", n_cells)
    rec.m("probe.cells_sha256", hashing.sha256_file(cells))
    rec.m("memory", mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ report
def summarise(rows: list[dict], tasks: list[str]) -> dict:
    """Q_text per (arm, seed, set, clf, budget) over draws; per-draw values kept for paired comparisons."""
    by: dict = {}
    for r in rows:
        k = (r["arm"], r["seed"], r["set"], r["clf"], str(r["budget"]), r["draw"])
        by.setdefault(k, {})[r["task"]] = r["acc"]
    q: dict = {}
    for (arm, seed, fs, clf, budget, draw), accs in by.items():
        if all(t in accs for t in tasks):
            q.setdefault((arm, seed, fs, clf, budget), {})[draw] = PR.q_text(accs, tasks)
    return q


def stage_report(args, p: dict, rec: Rec) -> Path:
    out = out_dir("report")
    rows = []
    for d in args.probe_runs.split(","):
        d = rp(d)
        with open(d / "cells.jsonl", encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    tasks = list(p["tasks"])
    q = summarise(rows, tasks)
    per_task: dict = {}
    for r in rows:
        per_task.setdefault((r["arm"], r["seed"], r["set"], r["clf"], str(r["budget"]), r["task"]), []).append(r["acc"])
    for k, v in per_task.items():
        rec.m("acc." + ".".join(str(x) for x in k), PR.t_interval(v), note="mean over draws, t interval")
    table = {}
    for k, draws in q.items():
        vals = [draws[d] for d in sorted(draws)]
        table[".".join(str(x) for x in k)] = PR.t_interval(vals)
        rec.m("q_text." + ".".join(str(x) for x in k), PR.t_interval(vals), note=f"draws {sorted(draws)}")
    def diff(a: dict, b: dict) -> dict:
        common = sorted(set(a) & set(b))
        return PR.paired_bootstrap(np.array([a[d] - b[d] for d in common])) if common else {"mean": None}

    arms = sorted({(k[0], k[1]) for k in q if k[2] in ARM_SETS})
    ranking = []
    for arm, seed in arms:
        for clf in ("lr", "tm"):
            for budget in [str(b) for b in p["budgets"]]:
                for var in ("all", "dim"):
                    bank = q.get((arm, seed, f"bank.{var}", clf, budget))
                    if not bank:
                        continue
                    c0y = q.get((arm, seed, f"c0y.{var}", clf, budget), {})
                    gp = q.get((arm, seed, f"gplus.{var}", clf, budget), {})
                    pre = f"compare.{arm}.{seed}.{clf}.{budget}.{var}"
                    rec.m(f"{pre}.bank_minus_c0y", diff(bank, c0y))
                    rec.m(f"{pre}.bank_minus_gplus", diff(bank, gp))
                    if c0y and gp:
                        strong = c0y if np.mean(list(c0y.values())) >= np.mean(list(gp.values())) else gp
                        rec.m(f"{pre}.bank_minus_max_c0y_gplus", diff(bank, strong),
                              note="per draw against the control with the higher mean Q_text")
                    bw = next((v for k, v in q.items() if k[2] == "bow" and k[3] == clf and k[4] == budget), None)
                    if bw:
                        rec.m(f"{pre}.bank_minus_bow", diff(bank, bw))
                    if clf == "lr" and budget == "32":
                        ranking.append({"arm": arm, "seed": seed, "variant": var, "q_text_mean": float(np.mean(list(bank.values())))})
    ranking.sort(key=lambda r: -r["q_text_mean"])
    rec.m("tournament.ranking_lr_32", ranking,
          note="exploratory ordering by mean Q_text at 32 labels per class (LR); A7 ties within 0.5 points go to Q_con, not computed here")
    (out / "q_text_table.json").write_text(json.dumps(table, indent=1, sort_keys=True))
    rec.flush()
    return out


# ------------------------------------------------------------------ data stage
def stage_data(args, p: dict, rec: Rec) -> None:
    for task in p["tasks"]:
        if args.smoke:
            v = D.verify_train_file(task)
            rec.check(f"data.{task}.train_sha256_matches_hash_file", v["equal"], **v)
            continue
        r = D.write_hash_file(task)
        rec.check(f"data.{task}.hash_file_equal", r["equal"], path=r["path"], written=r["written"])
        rec.m(f"data.{task}.hash_file", r)
        dataset_hashes({f"{task}_train": D.verify_train_file(task)["sha256"]})
    rec.flush()


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "train", "features", "baselines", "probe", "report", "all"])
    ap.add_argument("--config", default=str(ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_pretext.json"))
    ap.add_argument("--arm", default="A1", choices=sorted(P.ARMS))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--engine", default="cuda", choices=["cuda", "cpu", "oracle"])
    ap.add_argument("--train-run", default=None)
    ap.add_argument("--features-run", default=None)
    ap.add_argument("--baselines-run", default=None)
    ap.add_argument("--probe-runs", default=None)
    ap.add_argument("--sets", default=",".join(ARM_SETS))
    ap.add_argument("--classifiers", default="lr,tm")
    ap.add_argument("--full", action="store_true", help="also the full-pool budget (costly; not in the default launch)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    p = load_params(args.config)
    if args.smoke != bool(p.get("smoke", False)):
        raise SystemExit("--smoke must be given exactly with a smoke config (params.smoke = true)")
    rec = Rec(args.smoke)
    rec.m("stage", args.stage)
    rec.m("config_params", p)
    if args.stage == "data":
        stage_data(args, p, rec)
    elif args.stage == "train":
        stage_train(args, p, rec)
    elif args.stage == "features":
        stage_features(args, p, rec, rp(args.train_run))
    elif args.stage == "baselines":
        stage_baselines(args, p, rec, rp(args.features_run))
    elif args.stage == "probe":
        stage_probe(args, p, rec, rp(args.features_run), rp(args.baselines_run))
    elif args.stage == "report":
        stage_report(args, p, rec)
    else:  # all (smoke only): every stage in this run directory
        if not args.smoke:
            raise SystemExit("--stage all is for the smoke only")
        stage_data(args, p, rec)
        run = out_dir("all")
        stage_train(args, p, rec)
        stage_features(args, p, rec, run)
        stage_baselines(args, p, rec, run)
        args.sets = ",".join(ARM_SETS + BASELINE_SETS)
        stage_probe(args, p, rec, run, run)
        args.probe_runs = str(run)
        stage_report(args, p, rec)
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
