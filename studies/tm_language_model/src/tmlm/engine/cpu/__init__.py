"""C++17 + OpenMP host engine behind a plain C ABI, loaded with ctypes (M3).

Normative text: docs/03 'Implementations, in build order'; docs/04 M3.

Modules: build.py (g++ -O3 -march=native -fopenmp -shared -fPIC, debug build with -fsanitize=address,undefined;
no cmake), engine.cpp (the host engine), cpu_engine.py (the EngineBase subclass). Bit-exact with the oracle and the
CUDA engine (tests/test_cpu_engine*.py, tests/test_cuda_*.py).
"""
