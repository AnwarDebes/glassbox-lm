"""Build the CUDA engine's kernels with plain nvcc into an sm_70 CUBIN (never PTX; the research protocol, "Hardware").

    python -m tmlm.engine.cuda.build            # release: -O3 -lineinfo
    python -m tmlm.engine.cuda.build --debug    # -G (device debug info, for compute-sanitizer source lines)
    python -m tmlm.engine.cuda.build --clean

The cubin is written next to this file as `kernels.cubin` (release) or `kernels_debug.cubin`.
`cuda_engine.py` rebuilds automatically when the sources (kernels.cu, core/tm_core.h, this file)
are newer than the cubin. The module is loaded with `cupy.RawModule(path=...)`, so CuPy never
compiles or JITs anything: the exact flags below are the whole toolchain.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core" / "tm_core.h"
SRC = HERE / "kernels.cu"
ARCH = "sm_70"  # Tesla V100, compute capability 7.0 (the research protocol, "Hardware": cubin only, never PTX)


def cubin_path(debug: bool = False) -> Path:
    return HERE / ("kernels_debug.cubin" if debug else "kernels.cubin")


def needs_build(debug: bool = False) -> bool:
    cubin = cubin_path(debug)
    if not cubin.exists():
        return True
    newest = max(CORE.stat().st_mtime, SRC.stat().st_mtime, Path(__file__).stat().st_mtime)
    return cubin.stat().st_mtime < newest


def find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
            if root and (Path(root) / "bin" / "nvcc").exists():
                return str(Path(root) / "bin" / "nvcc")
        raise RuntimeError("nvcc not found (source env/env.sh)")
    return nvcc


def build_flags(debug: bool = False) -> list[str]:
    flags = ["-cubin", f"-arch={ARCH}", "-std=c++17", "-Wno-deprecated-gpu-targets",
             "-Xcompiler", "-Wall"]
    if debug:
        flags += ["-G"]
    else:
        flags += ["-O3", "-lineinfo", "-DNDEBUG"]
    return flags


def build(debug: bool = False, verbose: bool = True) -> Path:
    nvcc = find_nvcc()
    cubin = cubin_path(debug)
    cmd = [nvcc, *build_flags(debug), str(SRC), "-o", str(cubin)]
    if verbose:
        print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return cubin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)
    if args.clean:
        for p in (cubin_path(False), cubin_path(True)):
            if p.exists():
                os.remove(p)
        return 0
    build(debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
