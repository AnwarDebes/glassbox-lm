#!/usr/bin/env python
"""Ops launcher for the 3c chain's GPU steps that import c3_common (2026-10-21; operational, no rule change).

    python -I -B c3_cuda_first.py SCRIPT [ARGS...]

Why: w1/build_corpus_c4.py sets os.environ["CUDA_VISIBLE_DEVICES"] = "" when it is imported (it was written for
CPU-only corpus building), and c3_common.py imports it, so c3_filter.py and c3_nearcopy.py could not see the GPU (the
chain's STOPs "filter exit 1" at 2026-10-21T08:54:11Z and 09:33:36Z, "RuntimeError: No CUDA GPUs are available").
CUDA reads that variable once, when it is initialised, so this launcher initialises CUDA first, checks that a device is
there, and then runs SCRIPT as __main__ with sys.argv = [SCRIPT, ARGS...]. Nothing else changes: the step's code,
arguments and outputs are untouched.
"""

from __future__ import annotations

import runpy
import sys

import torch


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: c3_cuda_first.py SCRIPT [ARGS...]")
    try:
        torch.cuda.init()
    except RuntimeError as e:
        raise SystemExit(f"c3_cuda_first.py: CUDA could not be initialised ({e}); the step was not run") from None
    if not torch.cuda.is_available():
        raise SystemExit("c3_cuda_first.py: torch.cuda.is_available() is False after initialisation; the step was not run")
    n = torch.cuda.device_count()
    print(f"c3_cuda_first.py: CUDA initialised before the step's imports ({n} device(s): {torch.cuda.get_device_name(0)})",
          flush=True)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
