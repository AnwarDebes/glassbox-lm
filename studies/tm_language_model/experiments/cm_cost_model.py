"""CM: cost model with measured constants (docs/05 CM; docs/11 section 10; docs/13 H3.2, H3.3, H6.2).

Stage m4 (docs/04 M4 ACCEPT "measured throughput table for C in {8k, 64k, 1M}" and "CM constants"):
the tiny-LM task (experiments/engine_common.TinyLmTask data: the 5M-token WikiText-103 train
slice at BPE V = 2000, the frozen validation eval set with its hidden mask) trained for a fixed
number of batches at every clause count, on the CUDA engine (per-phase CUDA-event timing) and,
when --engines includes cpu, on the C++/OpenMP engine, under the EXCLUSIVE GPU lock:

    scripts/gpu_run.sh exclusive -- scripts/run_logged.py --name m4_cuda_throughput \\
        --config experiments/configs/CM.json -- .venv-tm/bin/python experiments/cm_cost_model.py --stage m4 \\
        [--clauses 8192,65536,1048576] [--batches 200] [--repeats 5] [--engines cuda,cpu] [--cpu-batches 50]
        [--engine-json experiments/configs/s4.json --engine-override '{"batch": 1, ...}' --predict-repeats 0]

Protocol per (engine, C): build the engine, train `--warmup` batches untimed (the state keeps
evolving), then `--repeats` timed blocks of `--batches` consecutive batches (the run continues
through the blocks, so block k sees the state after k blocks: the median and the min to max range
are over blocks of a run in progress, not over identical states), then `--repeats` timed passes of
predict_scores over the frozen validation targets. Reported: examples/s (training), positions/s
(predict), the per-phase split of the CUDA engine, and the constants of docs/11 section 10
derived from the phase times and the DIAG counters (ns per candidate, ns per included literal of
a candidate, ns per fired clause, ns per applied event, sweep ns per clause, bytes per clause
from the CuPy pool at two clause counts). Every number is written with its label; the load
average and the thread cap are recorded because the CPU quota is shared (research protocol, "Hardware").

Stage m7 (context length N with bags at every bucket count, streaming inference) is not part
of this file yet; it is written with s1 and s2 (docs/04 M7).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from engine_common import ROOT, EngineConfig, load_json, measured, top1, write_metrics_or_print

sys.path.insert(0, str(ROOT / "src"))
from tmlm.data.corpus_store import load_split  # noqa: E402
from tmlm.data.evalsets import load_evalset  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402

CORPUS = "wikitext103"


def derived(value, **extra) -> dict:
    d = {"value": value, "label": "derived"}
    d.update(extra)
    return d


def med_range(xs: list[float]) -> dict:
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def run_engine(kind: str, cfg: EngineConfig, train, val, targets, gold, hidden, batches: int, repeats: int, warmup: int,
               predict_repeats: int, block: int) -> dict:
    out: dict = {}
    t0 = time.time()
    if kind == "cuda":
        import cupy as cp
        from tmlm.engine.cuda.cuda_engine import CudaEngine
        pool = cp.get_default_memory_pool()
        before = pool.used_bytes()
        eng = CudaEngine(cfg, train, block=block, profile=True)
        eng.synchronize()
        out["device_bytes_after_build"] = int(pool.used_bytes() - before)
        sync = eng.synchronize
    else:
        eng = CpuEngine(cfg, train)
        sync = lambda: None  # noqa: E731
    out["build_s"] = time.time() - t0
    e = b = 0

    def step():
        nonlocal e, b
        d = eng.train_batch(e, b)
        b += 1
        if b >= eng.n_batches():
            e, b = e + 1, 0
        return d

    for _ in range(warmup):
        step()
    sync()
    if kind == "cuda":
        eng.reset_profile()
    blocks = []
    for r in range(repeats):
        diag: dict[str, int] = {}
        if kind == "cuda":
            eng.reset_profile()
        sync()
        t0 = time.time()
        for _ in range(batches):
            d = step()
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
        sync()
        wall = time.time() - t0
        blk = {"wall_s": wall, "examples": diag["n_examples"], "examples_per_s": diag["n_examples"] / wall, "diag": diag}
        if kind == "cuda":
            blk["phase_ms"] = dict(eng.profile_ms)
            blk["profile_counts"] = dict(eng.profile_counts)
            blk["device_bytes_used"] = int(cp.get_default_memory_pool().used_bytes())
            blk["device_bytes_total_pool"] = int(cp.get_default_memory_pool().total_bytes())
        blocks.append(blk)
        print(f"  {kind} C={cfg.n_clauses} block {r}: {blk['examples_per_s']:.0f} ex/s over {batches} batches, "
              f"cand/ex {diag['n_candidates'] / diag['n_examples']:.1f} fired/ex {diag['n_fired'] / diag['n_examples']:.2f} "
              f"events/ex {diag['n_events_applied'] / diag['n_examples']:.2f}", flush=True)
    out["train_blocks"] = blocks
    out["train_examples_per_s"] = med_range([x["examples_per_s"] for x in blocks])
    out["alive_after"] = int(eng.state()["alive"].sum())
    # prediction on the frozen validation targets through a probe engine over the validation split
    if predict_repeats <= 0:
        out["predict_blocks"], out["predict_positions_per_s"], out["top1_after_training"] = [], None, None
        del eng
        if kind == "cuda":
            cp.get_default_memory_pool().free_all_blocks()
        return out
    if kind == "cuda":
        probe = CudaEngine(cfg, val, block=block, profile=True)
    else:
        probe = CpuEngine(cfg, val)
    probe.load_state(eng.state())
    pred = []
    acc = None
    for r in range(predict_repeats):
        sync()
        t0 = time.time()
        acc = top1(probe, 4, targets, gold, hidden)
        sync()
        wall = time.time() - t0
        pred.append({"wall_s": wall, "positions": int(targets.shape[0]), "positions_per_s": targets.shape[0] / wall})
        if kind == "cuda":
            pred[-1]["phase_ms"] = dict(probe.profile_ms)
            probe.reset_profile()
        print(f"  {kind} C={cfg.n_clauses} predict {r}: {pred[-1]['positions_per_s']:.0f} pos/s, top1 {acc:.4f}", flush=True)
    out["predict_blocks"] = pred
    out["predict_positions_per_s"] = med_range([x["positions_per_s"] for x in pred]) if pred else None
    out["top1_after_training"] = acc
    del probe, eng
    if kind == "cuda":
        cp.get_default_memory_pool().free_all_blocks()
    return out


def constants_from(blocks: list[dict], cfg: EngineConfig) -> dict:
    """docs/11 section 10 constants from the CUDA phase times (ms) and DIAG sums of each block."""
    per = {"ns_per_candidate": [], "ns_per_candidate_included_literal": [], "ns_per_fired_clause_scores": [],
           "ns_per_applied_event": [], "ns_per_example_build_lookup": [], "sweep_ns_per_clause": [], "index_ns_per_clause": [],
           "ns_per_example_total": []}
    for blk in blocks:
        d, ph, pc = blk["diag"], blk["phase_ms"], blk["profile_counts"]
        n_ex = d["n_examples"]
        per["ns_per_candidate"].append(ph["verify"] * 1e6 / max(1, d["n_candidates"]))
        per["ns_per_candidate_included_literal"].append(ph["verify"] * 1e6 / max(1, pc["n_inc_of_candidates"]))
        per["ns_per_fired_clause_scores"].append(ph["scores"] * 1e6 / max(1, d["n_fired"]))
        per["ns_per_applied_event"].append(ph["feedback"] * 1e6 / max(1, d["n_events_applied"]))
        per["ns_per_example_build_lookup"].append((ph["build"] + ph["lookup"]) * 1e6 / max(1, n_ex))
        per["ns_per_example_total"].append(blk["wall_s"] * 1e9 / max(1, n_ex))
        if pc["sweeps"]:
            per["sweep_ns_per_clause"].append(ph["sweep"] * 1e6 / (pc["sweeps"] * cfg.n_clauses))
        per["index_ns_per_clause"].append(ph["index"] * 1e6 / (max(1, pc["batches"]) * cfg.n_clauses))
    return {k: med_range(v) for k, v in per.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "CM.json"))
    ap.add_argument("--stage", default="m4", choices=["m4"])
    ap.add_argument("--clauses", default="8192,65536,1048576")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--cpu-batches", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--predict-repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--engines", default="cuda,cpu")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine-json", default=str(ROOT / "experiments" / "configs" / "tiny_lm.json"),
                    help="run definition whose 'engine' block gives the base EngineConfig values")
    ap.add_argument("--engine-override", default="{}", help="JSON object merged over the engine block (e.g. the s4 rung flags)")
    ap.add_argument("--cpu-repeats", type=int, default=None, help="timed blocks for the CPU engine (default: --repeats)")
    args = ap.parse_args()
    cm_cfg = load_json(args.config)
    tiny = load_json(args.engine_json)
    eng_over = {**dict(tiny["engine"]), **json.loads(args.engine_override)}
    vocab = int(eng_over.get("vocab", 2000))
    from m3_tiny_lm import slice_train
    train = slice_train(load_split(CORPUS, vocab, "train"), args.tokens)
    val = load_split(CORPUS, vocab, "validation")
    ev = load_evalset(CORPUS, vocab, "validation", "subword")
    hidden = ev["hidden"].astype(np.uint8)
    targets = ev["targets"].astype(np.uint32)
    gold = val.tok[targets]
    metrics: dict = {
        "stage": measured(args.stage), "train_tokens": measured(int(train.tok.shape[0])), "val_targets": measured(int(targets.shape[0])),
        "evalset_sha256": measured(ev["meta"].get("sha256")), "batches_per_block": measured(args.batches),
        "cpu_batches_per_block": measured(args.cpu_batches), "repeats": measured(args.repeats), "warmup_batches": measured(args.warmup),
        "omp_num_threads": measured(os.environ.get("OMP_NUM_THREADS")), "loadavg_at_start": measured(list(os.getloadavg())),
        "cuda_block": measured(args.block), "engine_config_base": measured(eng_over),
        "protocol_note": measured("timed blocks are consecutive batches of one continuing run per (engine, C); median and min to max over blocks"),
    }
    print(f"train {train.tok.shape[0]} tokens, {targets.shape[0]} validation targets; loadavg {os.getloadavg()}", flush=True)
    sizes = [int(x) for x in args.clauses.split(",")]
    engines = [x for x in args.engines.split(",") if x]
    device_bytes: dict[int, int] = {}
    for kind in engines:
        for Cn in sizes:
            cfg = EngineConfig(**{**eng_over, "seed": args.seed, "vocab": vocab, "n_clauses": Cn})
            nb = args.batches if kind == "cuda" else args.cpu_batches
            reps = args.repeats if kind == "cuda" or args.cpu_repeats is None else args.cpu_repeats
            print(f"== {kind} C={Cn} ({nb} batches per block, {reps} blocks)", flush=True)
            res = run_engine(kind, cfg, train, val, targets, gold, hidden, nb, reps, args.warmup, args.predict_repeats, args.block)
            key = f"{kind}.C{Cn}"
            metrics[f"{key}.train_examples_per_s"] = measured(res["train_examples_per_s"], unit="examples/s")
            metrics[f"{key}.train_ms_per_batch"] = measured({k: 1000.0 * cfg.batch / v if k != "n" else v for k, v in res["train_examples_per_s"].items()}, unit="ms/batch", note="batch size / examples per s (median of blocks maps to median; min and max swap)")
            if res["predict_blocks"]:
                metrics[f"{key}.predict_positions_per_s"] = measured(res["predict_positions_per_s"], unit="positions/s")
                metrics[f"{key}.top1_after_training"] = measured(res["top1_after_training"], note="after warmup + repeats x batches, not the tiny-LM budget")
                metrics[f"{key}.predict_blocks"] = measured(res["predict_blocks"])
            metrics[f"{key}.alive_after"] = measured(res["alive_after"])
            metrics[f"{key}.build_s"] = measured(res["build_s"])
            metrics[f"{key}.batches_run"] = measured(args.warmup + reps * nb)
            metrics[f"{key}.blocks"] = measured([{k: v for k, v in blk.items()} for blk in res["train_blocks"]])
            if kind == "cuda":
                metrics[f"{key}.device_bytes_after_build"] = measured(res["device_bytes_after_build"], unit="bytes")
                device_bytes[Cn] = res["device_bytes_after_build"]
                metrics[f"{key}.phase_ms_median_per_block"] = measured({p: statistics.median([blk["phase_ms"][p] for blk in res["train_blocks"]]) for p in res["train_blocks"][0]["phase_ms"]})
                metrics[f"{key}.phase_ms_per_batch_median_block"] = measured({p: statistics.median([blk["phase_ms"][p] for blk in res["train_blocks"]]) / nb for p in res["train_blocks"][0]["phase_ms"]}, unit="ms/batch")
                for name, val_ in constants_from(res["train_blocks"], cfg).items():
                    metrics[f"CM.{key}.{name}"] = derived(val_, note="phase device time (CUDA events) over DIAG sums of the same block")
            os.environ.get("TMLM_RUN_DIR") and write_metrics_or_print(dict(metrics))
    if len(device_bytes) >= 2:
        cs = sorted(device_bytes)
        slope = (device_bytes[cs[-1]] - device_bytes[cs[0]]) / (cs[-1] - cs[0])
        metrics["CM.cuda.bytes_per_clause"] = derived(slope, note=f"(bytes at C={cs[-1]} minus bytes at C={cs[0]}) / (C difference), CuPy pool used bytes after build")
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
