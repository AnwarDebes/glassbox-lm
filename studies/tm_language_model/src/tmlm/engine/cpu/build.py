"""Build the C++ host engine with plain g++ (no cmake, no pybind11; docs/03 "Implementations").

    python -m tmlm.engine.cpu.build            # release: -O3 -march=native -fopenmp
    python -m tmlm.engine.cpu.build --debug    # -O1 -g -fsanitize=address,undefined
    python -m tmlm.engine.cpu.build --clean

The shared object is written next to this file as `libtmcpu.so` (release) or `libtmcpu_debug.so`.
`cpu_engine.py` rebuilds automatically when the sources are newer than the library.
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
SRC = HERE / "engine.cpp"


def lib_path(debug: bool = False) -> Path:
    return HERE / ("libtmcpu_debug.so" if debug else "libtmcpu.so")


def needs_build(debug: bool = False) -> bool:
    lib = lib_path(debug)
    if not lib.exists():
        return True
    newest = max(CORE.stat().st_mtime, SRC.stat().st_mtime, Path(__file__).stat().st_mtime)
    return lib.stat().st_mtime < newest


def build(debug: bool = False, verbose: bool = True) -> Path:
    gxx = shutil.which("g++")
    if gxx is None:
        raise RuntimeError("g++ not found on PATH")
    lib = lib_path(debug)
    flags = ["-std=c++17", "-Wall", "-Wextra", "-shared", "-fPIC", "-fopenmp"]
    if debug:
        flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    else:
        flags += ["-O3", "-march=native", "-DNDEBUG"]
    cmd = [gxx, *flags, str(SRC), "-o", str(lib)]
    if verbose:
        print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return lib


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)
    if args.clean:
        for p in (lib_path(False), lib_path(True)):
            if p.exists():
                os.remove(p)
        return 0
    build(debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
