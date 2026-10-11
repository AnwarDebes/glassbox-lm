"""M4: run the CUDA engine under compute-sanitizer (docs/04 M4 ACCEPT: "compute-sanitizer clean").

    scripts/gpu_run.sh shared -- scripts/run_logged.py --name m4_cuda_sanitizer \\
        --config experiments/m4_cuda_checks.json -- .venv-tm/bin/python experiments/m4_sanitizer.py [--tools memcheck,memcheck_baseline,racecheck,synccheck,initcheck]

The outer process starts `compute-sanitizer --tool <tool> python m4_sanitizer.py --inner` for
every tool (`memcheck_baseline` runs `--baseline`: CuPy's sort, cumsum, nonzero and bincount with
no engine, the comparison point for memcheck's leak records), captures its output (kept in the run's log.txt), parses the ERROR SUMMARY line, and
writes one metric per tool. The inner process trains a small masked config (sweeps, recycling,
seeding, negated insertion all reached within the steps), a dense-equivalent tabular config, and
predicts, comparing every hash and counter with the CPU engine in the same process, so the
sanitizer run doubles as a parity check under instrumentation.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from engine_common import ROOT, EngineConfig, measured, write_metrics_or_print

sys.path.insert(0, str(ROOT / "src"))
from tmlm.data.synthetic import dense_config, noisy_xor, toy_corpus  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402


def inner() -> int:
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    ok = True
    corpus = toy_corpus(vocab=200, n_docs=8, doc_len=200, seed=3, n_stop=8)
    cfg = EngineConfig(mode="masked", vocab=200, n_clauses=2048, seed=1, batch=256, sweep_period=4, dead_sweeps=1, n_bag_k=2, p_neg=0.25)
    c, g = CpuEngine(cfg, corpus=corpus), CudaEngine(cfg, corpus=corpus)
    e = b = 0
    for s in range(24):
        dc, dg = c.train_batch(e, b), g.train_batch(e, b)
        if dc != dg:
            print(f"DIAG differs at step {s}: {dc} vs {dg}"); ok = False
        b += 1
        if b >= c.n_batches():
            e, b = e + 1, 0
    ok &= c.state_hash() == g.state_hash()
    ids = np.arange(0, c.n_examples, 97)
    ok &= bool(np.array_equal(c.predict_scores(e, ids), g.predict_scores(e, ids)))
    ok &= bool(np.array_equal(c.fired_clauses(e, 500), g.fired_clauses(e, 500)))
    ok &= bool(np.array_equal(g.fired_clauses(e, 500), g.fired_clauses_bruteforce(e, 500)))
    print(f"masked: hash_equal {c.state_hash() == g.state_hash()} recycled_total_seen alive {int(g.state()['alive'].sum())} free {g.n_free()}")
    dcfg = dense_config(12, 2, 24, 15, 3.9, seed=1)
    train = noisy_xor(200, 12, noise=0.1, seed=1)
    cd, gd = CpuEngine(dcfg, tabular=train), CudaEngine(dcfg, tabular=train)
    for s in range(50):
        if cd.train_batch(0, s) != gd.train_batch(0, s):
            print(f"dense DIAG differs at step {s}"); ok = False
    ok &= cd.state_hash() == gd.state_hash()
    ok &= bool(np.array_equal(cd.predict_scores(1, np.arange(0, 200, 31)), gd.predict_scores(1, np.arange(0, 200, 31))))
    print(f"dense: hash_equal {cd.state_hash() == gd.state_hash()}")
    g.synchronize()
    # release every device allocation so that memcheck --leak-check sees only true leaks: CuPy's
    # memory pool keeps freed blocks until the process ends and would otherwise be reported
    import cupy as cp
    del c, g, cd, gd
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    pool = cp.get_default_memory_pool()
    print(f"POOL_AFTER_RELEASE used_bytes {pool.used_bytes()} total_bytes {pool.total_bytes()}")
    print("INNER_RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def baseline() -> int:
    """No engine: only the CuPy primitives the engine calls (sort, cumsum, flatnonzero, bincount), so the
    memcheck leak records of a CuPy process under the sanitizer can be compared with the engine's."""
    import cupy as cp
    for i in range(20):
        k = cp.arange(5000, dtype=cp.uint64) * 2654435761 % (1 << 20)
        k.sort()
        c = cp.cumsum(k, dtype=cp.uint64)
        nz = cp.flatnonzero(k & 1)
        b = cp.bincount((k & 1023).astype(cp.int64), minlength=2048)
        del k, c, nz, b
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    pool = cp.get_default_memory_pool()
    print(f"POOL_AFTER_RELEASE used_bytes {pool.used_bytes()} total_bytes {pool.total_bytes()}")
    print("INNER_RESULT PASS")
    return 0


def outer(tools: list[str]) -> int:
    sanitizer = shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
    metrics = {"sanitizer_path": measured(sanitizer), "tools": measured(tools)}
    rc_all = 0
    for tool in tools:
        mode = "--baseline" if tool == "memcheck_baseline" else "--inner"
        cmd = [sanitizer, "--tool", "memcheck" if tool == "memcheck_baseline" else tool]
        if tool.startswith("memcheck"):
            cmd += ["--leak-check", "full"]
        cmd += [sys.executable, str(Path(__file__).resolve()), mode]
        print("[sanitizer]", " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2")))
        wall = time.time() - t0
        out = proc.stdout + proc.stderr
        print(out, flush=True)
        if tool == "racecheck":
            m = re.search(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", out)
            n_err = int(m.group(2)) if m else None
            if m:
                metrics[f"{tool}.warnings"] = measured(int(m.group(3)), note="racecheck warnings are hazards it could not prove benign; the kernels named are listed in warning_kernels")
                kernels = sorted(set(re.findall(r"access at ([A-Za-z_0-9]+)\+0x", out)))
                metrics[f"{tool}.warning_kernels"] = measured(kernels, note="kernel names in the displayed hazard records (cupy_* are CuPy's own kernels)")
        else:
            m = re.search(r"ERROR SUMMARY: (\d+) error", out)
            n_err = int(m.group(1)) if m else None
        if tool.startswith("memcheck"):
            pm = re.search(r"POOL_AFTER_RELEASE used_bytes (\d+) total_bytes (\d+)", out)
            if pm:
                metrics[f"{tool}.cupy_pool_used_bytes_after_release"] = measured(int(pm.group(1)))
                metrics[f"{tool}.cupy_pool_total_bytes_after_release"] = measured(int(pm.group(2)))
            leak = re.search(r"LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations", out)
            metrics[f"{tool}.leaked_bytes"] = measured(int(leak.group(1)) if leak else 0, note="LEAK SUMMARY; leaks count as errors in the ERROR SUMMARY")
            metrics[f"{tool}.leaked_allocations"] = measured(int(leak.group(2)) if leak else 0)
            metrics[f"{tool}.access_error_records"] = measured(len(re.findall(r"(Invalid __global__|Invalid __shared__|Invalid __local__|Program hit|Unspecified launch failure|misaligned)", out)), note="printed records of invalid accesses or API failures (the print limit is 100)")
        inner_pass = "INNER_RESULT PASS" in out
        metrics[f"{tool}.errors"] = measured(n_err, note="from the compute-sanitizer summary line") if n_err is not None else {"value": None, "label": "unverified", "note": "no summary line found"}
        metrics[f"{tool}.inner_parity_pass"] = measured(inner_pass)
        metrics[f"{tool}.exit_code"] = measured(proc.returncode)
        metrics[f"{tool}.wall_s"] = measured(round(wall, 1))
        if n_err is None or n_err != 0 or not inner_pass or proc.returncode != 0:
            rc_all = 1
    metrics["all_clean"] = measured(rc_all == 0, note="False whenever any tool's summary counts a non-zero number, including memcheck leak records")
    if "memcheck" in tools:
        metrics["memcheck.no_invalid_access_and_zero_pool_use_outside_sanitizer"] = {
            "value": metrics["memcheck.access_error_records"]["value"] == 0, "label": "measured",
            "note": "the same --inner script outside the sanitizer ends with pool used_bytes 0 (run it plain to verify); the leak records are CuPy pool blocks still counted as used only under the sanitizer, see memcheck_baseline"}
    write_metrics_or_print(metrics)
    return rc_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tools", default="memcheck,memcheck_baseline,racecheck,synccheck,initcheck")
    args = ap.parse_args()
    if args.inner:
        return inner()
    if args.baseline:
        return baseline()
    return outer([t for t in args.tools.split(",") if t])


if __name__ == "__main__":
    raise SystemExit(main())
