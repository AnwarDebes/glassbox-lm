"""FOR THE GPU PHASE (not run in the CPU-only profile phase): kernel-level split of one CoTM training step on the V100.

Synthetic bits with the measured density of the real view-B bits (0.458, results/C1.json) at the shapes of a recorded
run: --shape abcd4k (4,000 clauses x 6,684 bits, s 16: e74c_abcd_Bc4_c4000_s16) | star16k (16,000 x 7,227, s 16) |
mwoz16k (16,000 x 6,580, s 8). Runs --warm steps, then --steps steps under torch.profiler (CPU + CUDA activity), and
prints the CUDA time per op class (binomial, GEMM, elementwise, rng) per step, with --impl orig (dl.tm.CoTM) or fast
(fast_tm.FastCoTM). The GPU-phase runner must hold the shared GPU lock and run alone (memory: GPU sharing slows TM
jobs). The same script runs on the CPU (--device cpu) as a code check.
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from fast_tm import FastCoTM  # noqa: E402

from dl.tm import CoTM, TMConfig  # noqa: E402

SHAPES = {"abcd4k": (4000, 6684, 16.0, 30), "star16k": (16000, 7227, 16.0, 30), "mwoz16k": (16000, 6580, 8.0, 128),
          "tiny": (64, 256, 16.0, 30)}


def classify(name: str) -> str:
    n = name.lower()
    if "binomial" in n:
        return "binomial"
    if "gemm" in n or "cutlass" in n or "aten::mm" in n or "aten::matmul" in n or "volta" in n or "xmma" in n:
        return "gemm"
    if "philox" in n or "uniform" in n or "random" in n or "randperm" in n or "rand" in n:
        return "rng"
    return "elementwise/other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="abcd4k", choices=sorted(SHAPES))
    ap.add_argument("--impl", default="orig", choices=["orig", "fast"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    N, f, s, C = SHAPES[args.shape]
    rng = np.random.default_rng(0)
    n = 128 * (args.warm + args.steps)
    X = (rng.random((n, f)) < 0.458).astype(np.uint8)
    y = rng.integers(0, C, n)
    cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=1, batch_size=128, seed=1)
    cls = CoTM if args.impl == "orig" else FastCoTM
    tm = cls(C, f, cfg, device=args.device)
    Xt = torch.as_tensor(X, device=args.device)
    yt = torch.as_tensor(y, device=args.device)
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    for i in range(args.warm):
        tm._step(Xt[128 * i : 128 * (i + 1)], yt[128 * i : 128 * (i + 1)])
    sync()
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.device == "cuda" else [])
    t0 = time.perf_counter()
    with profile(activities=acts) as prof:
        for i in range(args.warm, args.warm + args.steps):
            tm._step(Xt[128 * i : 128 * (i + 1)], yt[128 * i : 128 * (i + 1)])
        sync()
    wall = (time.perf_counter() - t0) / args.steps
    per = defaultdict(float)
    nk = 0
    for e in prof.events():
        if args.device == "cuda":
            if e.device_type == torch.autograd.DeviceType.CUDA:
                per[classify(e.name)] += e.device_time_total / 1e3 / args.steps  # ms per step (kernel time)
                nk += 1
        elif e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::"):
            per[classify(e.name)] += e.self_cpu_time_total / 1e3 / args.steps
    res = {"shape": args.shape, "impl": args.impl, "device": args.device, "N": N, "f": f, "s": s, "wall_ms_per_step": wall * 1e3,
           "kernel_ms_per_step_by_class": dict(per), "kernels_per_step": nk / args.steps if nk else None,
           "torch": torch.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
