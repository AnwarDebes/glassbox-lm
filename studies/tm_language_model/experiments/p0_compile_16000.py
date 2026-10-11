"""T0.3: do the STOCK GraphTsetlinMachine kernels compile and train with 16000 outputs at
the initial design's E3 scale (clauses 8192, hypervector 2048, depth 3, message_size 1024)?

`CLASSES` is a `#define` that the stock package injects when it builds its kernels
(docs/02: tm.py:437-450), so "16000 outputs" is a compile-time question. The kernels are
built on the first fit() call, not in the constructor. One attempt therefore is: build the
two-node task with a few graphs, construct the model, call fit(epochs=1, incremental=True)
once as warm-up (this is where nvcc runs), then a few more timed fit() calls and one
predict() (the inference kernels live in the same modules, but predict allocates its own
buffers). The pycuda compiler cache is DISABLED, so the logged build time is a real nvcc
build and not a cache hit.

Each attempt runs in a fresh subprocess (experiments/p0_common.py). If it fails, the exact
error text and the traceback are logged and stored in the run dir; a failure is a valid
outcome of this task and is reported as such.

Outputs in the run dir: compile_16000.json, workers/*.json, workers/*.log, metrics.json.

Run (a build time is reported, so the GPU lock is exclusive):
    python experiments/p0_compile_16000.py --write-config /path/cfg.json
    scripts/gpu_run.sh exclusive -- scripts/run_logged.py --name p0_compile_16000 \
        --config /path/cfg.json -- python experiments/p0_compile_16000.py --config /path/cfg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as common  # noqa: E402

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "p0_compile_16000",
    "brief_tasks": ["T0.3"],
    "package": "stock GraphTsetlinMachine (vendored master, unmodified)",
    "task": "two-node graphs, symbols A/B, label = hash(symbol pair) mod K plus label noise",
    "base": {
        "outputs": 16000,
        "clauses": 8192, "T": 1000, "s": 10.0, "q": 1.0, "max_included_literals": None,
        "depth": 3,
        "hypervector_size": 2048, "hypervector_bits": 2,
        "message_size": 1024, "message_bits": 2,
        "n_graphs": 64, "noise": 0.25, "seed": 42, "reps": 3,
        "grid": list(common.SAFE_GRID), "block": list(common.SAFE_BLOCK),
        "mode": "time",
        "disable_pycuda_cache": True,
        "score_after": True,
    },
    "attempts": [
        {"id": "multiclass", "model": "multiclass"},
        {"id": "multioutput", "model": "multioutput"},
    ],
    "worker_timeout_s": 3600,
}


def expected_bytes(base: dict[str, Any]) -> dict[str, int]:
    """Buffer sizes from the memory formulas of docs/02. DERIVED, not measured."""
    ta_chunks = (2 * base["hypervector_size"] - 1) // 32 + 1
    message_chunks = (2 * base["message_size"] - 1) // 32 + 1
    state_bits = 8
    return {
        "ta_state": base["clauses"] * ta_chunks * state_bits * 4,
        "message_ta_state": (base["depth"] - 1) * base["clauses"] * message_chunks
        * state_bits * 4,
        "clause_weights": base["outputs"] * base["clauses"] * 4,
        "class_clause_update": base["outputs"] * base["clauses"] * 4,
        "encoded_Y": base["n_graphs"] * base["outputs"] * 4,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="JSON config (default: DEFAULT_CONFIG)")
    parser.add_argument("--write-config", default=None,
                        help="write DEFAULT_CONFIG to this path and exit")
    args = parser.parse_args(argv)
    if args.write_config:
        Path(args.write_config).write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    config = common.load_config(args.config, DEFAULT_CONFIG)
    base = config["base"]

    with common.open_run("p0_compile_16000", config) as run:
        work_dir = run.run_dir / "workers"
        derived = expected_bytes(base)
        report: dict[str, Any] = {"config": config, "expected_bytes_derived": derived,
                                  "attempts": {}}
        n_failed = 0
        for attempt in config["attempts"]:
            tag = f"compile16000_{attempt['id']}"
            spec = {**base, "model": attempt["model"], "tag": tag}
            result = common.spawn_worker(spec, work_dir,
                                         timeout_s=float(config["worker_timeout_s"]))
            ok = "error" not in result
            report["attempts"][attempt["id"]] = result
            key = attempt["id"]
            run.add_metric(f"compile_and_train_success.{key}", bool(ok), "measured",
                           note="stock kernels built with CLASSES=16000 and fit() plus "
                                "predict() ran without an exception")
            if not ok:
                n_failed += 1
                run.add_metric(f"error.{key}", str(result["error"]), "measured")
                print(f"[result] {key}: FAILED: {result['error']}", flush=True)
                print(result.get("traceback") or result.get("log_tail", ""), flush=True)
                continue
            run.add_metric(f"kernel_build_s.{key}", result["init_gpu_kernels_s"], "measured",
                           unit="s", note="stock _init_gpu_kernels (3 nvcc builds), "
                                          "pycuda compiler cache disabled")
            run.add_metric(f"warmup_fit_s.{key}", result["warmup_fit_s"], "measured", unit="s",
                           note="first fit(): kernel build, allocation, uploads, "
                                f"{base['n_graphs']} examples")
            run.add_metric(f"number_of_outputs.{key}", result["number_of_outputs"], "measured")
            run.add_metric(f"gpu_alloc_mib_after_training.{key}",
                           result["gpu_alloc_mib_after_training"], "measured", unit="MiB",
                           note="sum of the sizes this process passed to pycuda mem_alloc "
                                "(model, graphs, dense label matrix)",
                           largest_buffer_mib=result["gpu_alloc_after_training"][
                               "largest_bytes"] / 2**20,
                           mem_alloc_calls=result["gpu_alloc_after_training"]["calls"])
            if "gpu_alloc_mib_after_predict" in result:
                run.add_metric(f"gpu_alloc_mib_after_predict.{key}",
                               result["gpu_alloc_mib_after_predict"], "measured", unit="MiB",
                               note="as above, plus the buffers predict() allocates")
            run.add_metric(f"gpu_context_only_mib.{key}", result["gpu_used_mib_context_only"],
                           "measured", unit="MiB",
                           note="DEVICE-WIDE, before the model exists: this context plus "
                                "any other process on the GPU")
            run.add_metric(f"gpu_used_after_training_mib.{key}",
                           result["gpu_used_mib_after_training"], "measured", unit="MiB",
                           note="DEVICE-WIDE, pycuda mem_get_info: includes the CUDA context "
                                "and any other process on the GPU")
            run.add_metric(f"gpu_model_and_data_mib.{key}", result["gpu_model_and_data_mib"],
                           "measured", unit="MiB",
                           note="device-wide used minus the context-only reading; wrong "
                                "whenever another process came or went in between")
            if "gpu_used_mib_after_predict" in result:
                run.add_metric(f"gpu_used_after_predict_mib.{key}",
                               result["gpu_used_mib_after_predict"], "measured", unit="MiB",
                               note="DEVICE-WIDE")
            run.add_metric(f"ms_per_example.{key}", result["ms_per_example"], "measured",
                           unit="ms", stat="median of timed fits",
                           n_graphs=base["n_graphs"], reps=result["fit_s"]["n"],
                           min=1000.0 * result["fit_s"]["min"] / base["n_graphs"],
                           max=1000.0 * result["fit_s"]["max"] / base["n_graphs"])
            run.add_metric(f"examples_per_s.{key}", result["examples_per_s"], "measured",
                           unit="examples/s", stat="median of timed fits")
            run.add_metric(f"label_matrix_mib.{key}", result["label_matrix_bytes"] / 2**20,
                           "measured", unit="MiB")
            if "train_accuracy" in result:
                run.add_metric(f"train_accuracy.{key}", result["train_accuracy"], "measured",
                               note="sanity only: a few epochs on a few noisy examples")
            print(f"[result] {key}: compiled and trained. kernel build "
                  f"{result['init_gpu_kernels_s']:.2f} s, "
                  f"{result['ms_per_example']:.2f} ms/example, GPU buffers allocated "
                  f"{result['gpu_alloc_mib_after_training']:.0f} MiB", flush=True)
        for name, value in derived.items():
            run.add_metric(f"expected_mib.{name}", value / 2**20, "derived", unit="MiB",
                           note="memory formulas of docs/02")
        run.write_json("compile_16000.json", report)
        print(f"[done] {len(config['attempts']) - n_failed} attempts ok, {n_failed} failed",
              flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
