"""E3: WikiText-103 masked-token LM at V = 8000 (docs/05 E3; docs/06; docs/13 H1.2, amendment A2; docs/04 M6
Gate 1). Config `experiments/configs/E3_v8k.json`: masked, V = 8000, C = 1,048,576, T = 64, s = 128, B = 1024,
13 bags, mask rate 0.15. Three modes, all on the CUDA engine under the shared GPU lock:

    # exploratory slice check (seed 101, never a result): the tiny-LM task at V = 8000, C = 65536, 2 and 13 bags
    scripts/gpu_run.sh shared -- scripts/run_logged.py --name E3_v8k_slice_exploratory --config experiments/configs/E3_v8k.json \\
        -- .venv-tm/bin/python -u experiments/e3_wikitext_masked.py --slice-tokens 5000000 --seed 101 --clauses 65536 --n-bag-k 2,13

    # measure first: build cost and a 200-batch probe at the full config, no epoch, no test access
    scripts/gpu_run.sh shared -- scripts/run_logged.py --name E3_v8k_probe --config experiments/configs/E3_v8k.json \\
        -- .venv-tm/bin/python -u experiments/e3_wikitext_masked.py --seed 1 --probe-batches 200

    # the pre-registered run, one seed per logged run
    scripts/gpu_run.sh shared -- scripts/run_logged.py --name E3_v8k --config experiments/configs/E3_v8k.json \\
        -- .venv-tm/bin/python -u experiments/e3_wikitext_masked.py --seed 1 [--epochs 4] [--resume ckpt.npz]

Protocol (docs/05 E3, docs/06): masked training over every train position (15% of the context hidden per
epoch, the target always hidden); after every epoch the FROZEN validation eval sets with their stored hidden
bitmaps: subword variant top-1 and top-5, whole-word variant piece top-1, piece top-5 and word accuracy (a
word counts when every piece is right, pieces predicted independently, as M2's baselines report it);
comparators are the M2 baseline logs at V = 8000 (left bigram, bidirectional KN-5, CBOW-kNN, BERT), written
next to the numbers with their run directories. Compressed checkpoint after every epoch (resumable), GPU
yield to exclusive jobs, then the TEST eval sets ONCE through the counted path after the validation metrics
are on disk. Gate 1 / H1.2: above the left bigram and CBOW-kNN on the frozen subword set, 3 seeds.
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

from engine_common import ROOT, EngineConfig, load_json

sys.path.insert(0, str(ROOT / "src"))
from tmlm.common import runlog  # noqa: E402
from tmlm.common.locks import yield_gpu_if_requested  # noqa: E402
from tmlm.data.corpus_store import load_split, split_dir  # noqa: E402
from tmlm.data.evalsets import load_evalset, test_access_count  # noqa: E402
from tmlm.engine.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from tmlm.eval.baselines.ngram import word_accuracy  # noqa: E402

CORPUS = "wikitext103"
METRICS: dict[str, dict] = {}
T0 = time.time()
# M2 comparators at V = 8000 on the frozen validation sets (visible context = the hidden mask applied)
COMPARATORS_V8000 = {
    "left_bigram.validation_subword.top1_visible": (0.1487, "results/20260922T180823Z_M2_baselines_ngram_v8000/"),
    "left_bigram.validation_subword.top1_truectx": (0.1687, "results/20260922T180823Z_M2_baselines_ngram_v8000/"),
    "kn5_both.validation_subword.top1_visible": (0.442, "results/20260922T180823Z_M2_baselines_ngram_v8000/ (bidirectional product of experts)"),
    "cbow_knn.validation_subword.top1_visible": (0.0936, "results/20260922T210329Z_M2_baselines_cbow_knn_v8000/"),
    "count_probe_masked.validation_subword.top1_visible": (0.2012, "results/20260922T192647Z_M2_baselines_count_probe_v8000/"),
    "bert.validation_wholeword.word_acc": (0.5401, "results/20260922T183153Z_M2_baselines_bert/ (contaminated reference)"),
}


def log(msg: str) -> None:
    print(f"[e3 {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def metric(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None:
        runlog.write_metrics({name: METRICS[name]})
    log(f"{name} = {value}")


def dataset_hash(name: str, digest: str) -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({name: digest}, on_conflict="replace")


def out_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "e3_unlogged"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def contiguous(corpus):
    from tmlm.engine.api import CorpusArrays
    return CorpusArrays(**{k: np.ascontiguousarray(getattr(corpus, k)) for k in ("tok", "doc_start", "kind_map", "stop", "sym_freq", "targets")})


# ------------------------------------------------------------------ masked evaluation on a frozen set
def topk_hits(engine, targets: np.ndarray, hidden: np.ndarray | None, gold: np.ndarray, k: int = 5, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Per-target top-1 prediction and top-k hit flag from the class sums (ties toward the lower id)."""
    pred = np.zeros(targets.shape[0], dtype=np.int64)
    hit_k = np.zeros(targets.shape[0], dtype=bool)
    for lo in range(0, targets.shape[0], chunk):
        pos = targets[lo:lo + chunk]
        s = engine.predict_scores(0, pos, hidden)
        pred[lo:lo + chunk] = s.argmax(axis=1)
        order = np.argsort(-s, axis=1, kind="stable")[:, :k]
        hit_k[lo:lo + chunk] = (order == gold[lo:lo + chunk, None]).any(axis=1)
    return pred, hit_k


def masked_eval(engine, split_tok: np.ndarray, sub: dict, ww: dict, prefix: str, note: str) -> dict:
    """The four docs/06 numbers plus top-5 on the frozen sets of one split."""
    out = {}
    t_sub = sub["targets"].astype(np.uint32)
    gold_s = split_tok[t_sub].astype(np.int64)
    t0 = time.time()
    p, h5 = topk_hits(engine, t_sub, sub["hidden"].astype(np.uint8), gold_s)
    out["subword.top1_visible"] = float((p == gold_s).mean())
    out["subword.top5_visible"] = float(h5.mean())
    out["subword.eval_s"] = time.time() - t0
    t_ww = ww["targets"].astype(np.uint32)
    gold_w = split_tok[t_ww].astype(np.int64)
    t0 = time.time()
    p, h5 = topk_hits(engine, t_ww, ww["hidden"].astype(np.uint8), gold_w)
    out["wholeword.piece_top1_visible"] = float((p == gold_w).mean())
    out["wholeword.piece_top5_visible"] = float(h5.mean())
    out["wholeword.word_acc_visible"] = word_accuracy(p, gold_w, ww["word_id"][t_ww])
    out["wholeword.eval_s"] = time.time() - t0
    for k, v in out.items():
        metric(f"{prefix}.{k}", v, note=note)
    return out


def write_comparators() -> None:
    for name, (v, src) in COMPARATORS_V8000.items():
        metric(f"comparator.{name}", v, note=f"from {src}")


# ------------------------------------------------------------------ training helpers
def train_epoch(engine, e: int, n_batches: int, b0: int, args, pool, scratch_ckpt: Path | None) -> tuple[dict, float]:
    diag: dict[str, int] = {}
    t_epoch = time.time()
    for b in range(b0, n_batches):
        d = engine.train_batch(e, b)
        for k, v in d.items():
            diag[k] = diag.get(k, 0) + v
        done = b + 1
        if done % args.log_every == 0 or done == n_batches:
            engine.synchronize()
            el = time.time() - t_epoch
            n_ex = diag["n_examples"]
            log(f"epoch {e + 1} batch {done}/{n_batches}: {n_ex / el:.0f} ex/s cand/ex {diag['n_candidates'] / n_ex:.1f} fired/ex {diag['n_fired'] / n_ex:.2f} "
                f"events/ex {diag['n_events_applied'] / n_ex:.2f} seeded {diag['n_seeded']} recycled {diag['n_recycled']} pool {pool.used_bytes() / 1e9:.2f} GB rss {rss_gib():.1f} GiB")
        if scratch_ckpt is not None and args.checkpoint_every and done % args.checkpoint_every == 0 and done < n_batches:
            engine.synchronize()
            save_checkpoint(scratch_ckpt, engine, e, done)
        if args.yield_every and done % args.yield_every == 0:
            engine.synchronize()
            if yield_gpu_if_requested():
                log("yielded the GPU to an exclusive job and took the shared lock back")
    engine.synchronize()
    return diag, time.time() - t_epoch


def report_epoch(p: str, diag: dict, wall: float, engine, pool, partial: str) -> None:
    n_ex = diag["n_examples"]
    metric(f"{p}.train_wall_s", wall, note="training only" + partial)
    metric(f"{p}.examples_per_s", n_ex / wall, unit="examples/s")
    metric(f"{p}.n_examples", n_ex)
    for name in ("n_candidates", "n_fired", "n_events_applied", "n_events_dropped_cap", "n_tgt_gate_pass", "n_neg_gate_pass", "n_inserted", "n_neg_inserted", "n_vetoes"):
        metric(f"{p}.{name}_per_example", diag[name] / n_ex)
    metric(f"{p}.n_seeded", diag["n_seeded"])
    metric(f"{p}.n_recycled", diag["n_recycled"])
    metric(f"{p}.alive_clauses", int(engine.state()["alive"].sum()))
    metric(f"{p}.device_pool_bytes", int(pool.used_bytes()), unit="bytes")
    metric(f"{p}.host_maxrss_gib", rss_gib(), unit="GiB")


# ------------------------------------------------------------------ modes
def slice_mode(args, base: dict) -> int:
    """Exploratory: the tiny-LM task at this vocabulary (TinyLmTask: slice, frozen subword set, left bigram and
    count probe on the same positions), trained on the CUDA engine per bag setting."""
    import cupy as cp
    from engine_common import TinyLmTask
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    vocab = int(base["vocab"])
    task = TinyLmTask(vocab=vocab, n_tokens=args.slice_tokens, mode="masked")
    for k, v in task.metrics().items():
        metric(f"slice.{k}", v["value"], note=v.get("note", "tiny-LM task at V = %d" % vocab))
    ww = load_evalset(CORPUS, vocab, "validation", "wholeword")
    sub = {"targets": task.targets, "hidden": task.hidden}
    write_comparators()
    metric("slice.note", "EXPLORATORY (seed %d): the tiny-LM slice at V = %d; never cited as a result" % (args.seed, vocab), label="derived")
    pool = cp.get_default_memory_pool()
    for nb in [int(x) for x in args.n_bag_k.split(",")]:
        cfg = EngineConfig(**{**base, "seed": args.seed, "n_clauses": args.clauses, "n_bag_k": nb})
        tag = f"slice.bags{nb}"
        metric(f"{tag}.engine_config", cfg.to_dict())
        eng = CudaEngine(cfg, task.train)
        probe = CudaEngine(cfg, task.val)
        n_batches = eng.n_batches()
        best = 0.0
        for e in range(args.epochs):
            diag, wall = train_epoch(eng, e, n_batches, 0, args, pool, None)
            p = f"{tag}.epoch{e + 1}"
            report_epoch(p, diag, wall, eng, pool, "")
            probe.load_state(eng.state())
            res = masked_eval(probe, task.val.tok, sub, ww, f"{p}.validation", f"frozen validation sets, epoch {e + 1}, exploratory")
            best = max(best, res["subword.top1_visible"])
            log(f"bags {nb} epoch {e + 1}: top1 {res['subword.top1_visible']:.4f} top5 {res['subword.top5_visible']:.4f} word_acc {res['wholeword.word_acc_visible']:.4f} "
                f"(bigram {task.left_bigram_top1:.4f}, count probe {task.count_probe_top1:.4f}) wall {wall:.0f} s")
        metric(f"{tag}.best_top1", best)
        metric(f"{tag}.beats_slice_left_bigram", bool(best > task.left_bigram_top1), note="left bigram with the true left token, counts from the slice (the tiny-LM comparator)")
        metric(f"{tag}.ratio_to_count_probe", best / max(1e-9, task.count_probe_top1), label="derived")
        del eng, probe
        pool.free_all_blocks()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "E3_v8k.json"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=4, help="docs/13 amendment A2")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--probe-batches", type=int, default=0)
    ap.add_argument("--slice-tokens", type=int, default=0, help="exploratory slice mode (tiny-LM task at this vocabulary)")
    ap.add_argument("--clauses", type=int, default=65536, help="slice mode only")
    ap.add_argument("--n-bag-k", default="2,13", help="slice mode only, comma separated")
    ap.add_argument("--checkpoint-every", type=int, default=20000)
    ap.add_argument("--scratch-dir", default=os.environ.get("TMLM_SCRATCH", "/tmp"))
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--yield-every", type=int, default=500)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    cfg_json = load_json(args.config)
    base = dict(cfg_json["engine"])
    vocab = int(base["vocab"])
    metric("seed", args.seed)
    metric("test_accesses_before", test_access_count())
    if args.slice_tokens:
        return slice_mode(args, base)

    cfg = EngineConfig(**{**base, "seed": args.seed})
    if cfg.mode != "masked":
        raise ValueError("E3 needs masked mode")
    metric("engine_config", cfg.to_dict())
    metric("epochs_planned", args.epochs, note="docs/13 amendment A2")
    t0 = time.time()
    train = contiguous(load_split(CORPUS, vocab, "train"))
    val = contiguous(load_split(CORPUS, vocab, "validation"))
    for split in ("train", "validation"):
        meta = json.loads((split_dir(CORPUS, vocab, split) / "meta.json").read_text())
        dataset_hash(f"{split}_tok_V{vocab}", meta["sha256"]["tok.npy"])
    sub = load_evalset(CORPUS, vocab, "validation", "subword")
    ww = load_evalset(CORPUS, vocab, "validation", "wholeword")
    for name, ev in (("subword", sub), ("wholeword", ww)):
        dataset_hash(f"evalset_validation_{name}_V{vocab}", ev["meta"]["sha256_combined"])
    metric("data_load_s", time.time() - t0)
    metric("n_train_tokens", int(train.tok.shape[0]))
    metric("n_validation_subword_targets", int(sub["targets"].shape[0]))
    metric("n_validation_wholeword_targets", int(ww["targets"].shape[0]))
    write_comparators()

    import cupy as cp
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    pool = cp.get_default_memory_pool()
    t0 = time.time()
    engine = CudaEngine(cfg, train, profile=bool(args.probe_batches))
    engine.synchronize()
    metric("engine_build_s", time.time() - t0)
    metric("device_pool_bytes_after_build", int(pool.used_bytes()), unit="bytes")
    metric("host_maxrss_gib_after_build", rss_gib(), unit="GiB")
    n_batches = engine.n_batches()
    metric("batches_per_epoch", n_batches)
    val_engine = None

    def probe_engine():
        nonlocal val_engine
        if val_engine is None:
            val_engine = CudaEngine(cfg, val)
        val_engine.load_state(engine.state())
        return val_engine

    if args.probe_batches:
        nb = args.probe_batches
        for b in range(5):
            engine.train_batch(0, b)
        engine.synchronize()
        engine.reset_profile()
        diag: dict[str, int] = {}
        t0 = time.time()
        for b in range(5, 5 + nb):
            d = engine.train_batch(0, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
        engine.synchronize()
        wall = time.time() - t0
        n_ex = diag["n_examples"]
        metric("probe.batches", nb)
        metric("probe.examples_per_s", n_ex / wall, unit="examples/s")
        metric("probe.ms_per_batch", 1000 * wall / nb)
        metric("probe.epoch_estimate_s_young_state", n_batches * wall / nb, label="derived", note="batches per epoch x ms per batch on a young state; the CP anchor run's epoch 1 ran at 0.33x its probe rate once candidates per example grew 40x (results/20260923T042350Z_CP_anchor/)")
        metric("probe.epoch_estimate_s_cp_scaled", n_batches * wall / nb / 0.33, label="derived", note="the young-state estimate divided by the CP run's measured epoch-1 / probe rate ratio 11,317 / 34,458 = 0.33")
        metric("probe.candidates_per_example", diag["n_candidates"] / n_ex)
        metric("probe.fired_per_example", diag["n_fired"] / n_ex)
        metric("probe.events_per_example", diag["n_events_applied"] / n_ex)
        metric("probe.phase_ms_per_batch", {k: v / nb for k, v in engine.profile_ms.items()})
        metric("probe.device_pool_bytes_after_training", int(pool.used_bytes()), unit="bytes")
        pe = probe_engine()
        t0 = time.time()
        res = masked_eval(pe, val.tok, sub, ww, "probe.validation", "after 5 + probe batches only, NOT a result")
        metric("probe.validation_eval_s", time.time() - t0)
        log("probe done")
        return 0

    start_epoch, start_batch = 0, 0
    if args.resume:
        ccfg, state, meta = load_checkpoint(args.resume)
        if ccfg != cfg:
            raise ValueError("checkpoint config differs from the run config")
        engine.load_state(state)
        start_epoch, start_batch = int(meta["position"]["epoch"]), int(meta["position"]["batch_index"])
        metric("resumed_from", {"path": str(args.resume), "state_hash": meta["state_hash"], "epoch": start_epoch, "batch_index": start_batch, "step": meta["step"]})
    scratch_ckpt = Path(args.scratch_dir) / f"e3_v{vocab}_seed{args.seed}_scratch_checkpoint.npz"
    trained = False
    for e in range(start_epoch, args.epochs):
        b0 = start_batch if e == start_epoch else 0
        diag, wall = train_epoch(engine, e, n_batches, b0, args, pool, scratch_ckpt)
        p = f"epoch{e + 1}"
        report_epoch(p, diag, wall, engine, pool, " (resumed mid-epoch: counters cover this process only)" if b0 else "")
        t0 = time.time()
        ck = out_dir() / f"checkpoint_epoch{e + 1}.npz"
        h = save_checkpoint(ck, engine, e + 1, 0, compress=True)
        metric(f"{p}.checkpoint", {"path": str(ck), "state_hash": h, "bytes": ck.stat().st_size, "write_s": time.time() - t0})
        pe = probe_engine()
        res = masked_eval(pe, val.tok, sub, ww, f"{p}.validation", f"frozen validation sets, epoch {e + 1}")
        log(f"epoch {e + 1} done: subword top1 {res['subword.top1_visible']:.4f} top5 {res['subword.top5_visible']:.4f} word_acc {res['wholeword.word_acc_visible']:.4f} wall {wall:.0f} s")
        trained = True
    if scratch_ckpt.exists():
        scratch_ckpt.unlink()
    if not trained:
        return 0
    last = f"epoch{args.epochs}.validation"
    top1 = METRICS[f"{last}.subword.top1_visible"]["value"]
    metric("H1_2_seed_validation_above_left_bigram_and_cbow_knn",
           bool(top1 > COMPARATORS_V8000["left_bigram.validation_subword.top1_visible"][0] and top1 > COMPARATORS_V8000["cbow_knn.validation_subword.top1_visible"][0]),
           note="validation, frozen subword set, visible context; H1.2 needs 3 seeds and the test set")
    if args.no_test:
        return 0
    purpose = f"E3 V={vocab} seed {args.seed}: final frozen test eval sets after {args.epochs} epochs, choices frozen on validation"
    metric("test_access_purpose", purpose)
    test = contiguous(load_split(CORPUS, vocab, "test", purpose=purpose))
    tsub = load_evalset(CORPUS, vocab, "test", "subword", purpose=purpose)
    tww = load_evalset(CORPUS, vocab, "test", "wholeword", purpose=purpose)
    metric("test_accesses_after", test_access_count())
    te = CudaEngine(cfg, test)
    te.load_state(engine.state())
    res = masked_eval(te, test.tok, tsub, tww, "test", "TEST, once, frozen test eval sets")
    metric("test_8gram_overlap_with_train", 0.114, note="docs/12 F16, results/20260922T160257Z_M2_data/")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
