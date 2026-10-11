"""The shared C header compiled by both g++ and nvcc: all feedback logic of docs/11 lives here once (M3, M4).

Normative text: docs/11 (normative); src/tmlm/engine/config.py (the Python side of the same constants).

Contents (no Python module here; the header is found by path, not imported):
- tm_core.h: constants, state layout, and the per-clause feedback routines; compiled into the CPU engine by
  engine/cpu/build.py and into the CUDA engine by engine/cuda/build.py (both locate it as core/tm_core.h);
  tests/test_tm_core_constants.py keeps it equal to config.py
"""
