"""CUDA engine: docs/11 on the one V100 (docs/03 kernels K1..K14; docs/04 M4; docs/11 section 9 sharding).

Modules:
- kernels.cu      the kernels, sharing core/tm_core.h with the CPU engine (all learning logic lives there)
- build.py        plain nvcc build into an sm_70 CUBIN (never PTX), loaded by cupy.RawModule(path=...)
- cuda_engine.py  CudaEngine(EngineBase): train_batch (DIAG counters), predict_scores, fired_clauses,
                  state, load_state, state_hash; counter-based randomness only, no curand, no float
                  arithmetic, results independent of grid and block sizes
- shard.py        planned: the s5 exchange step of docs/11 section 9 (not written at M4)

Checkpoint and resume for every engine: tmlm.engine.checkpoint.
Parity with the CPU engine: tests/test_cuda_engine_parity.py; replay: tests/test_engine_checkpoint.py.
"""
