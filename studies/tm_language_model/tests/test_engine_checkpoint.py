"""docs/04 M4 "checkpoint and resume": a run interrupted at any batch and resumed from its
checkpoint reproduces the hash of the uninterrupted run (docs/13 H2.2 replay), on the CPU engine,
on the CUDA engine, and across the two (CPU checkpoint resumed on the GPU and back)."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import toy_corpus
from tmlm.engine.checkpoint import load_checkpoint, next_position, save_checkpoint, train_to
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine

CFG = dict(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2)


def _corpus():
    return toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)


def _uninterrupted(engine_cls, corpus, n_steps: int) -> tuple[str, tuple[int, int]]:
    eng = engine_cls(EngineConfig(**CFG), corpus=corpus)
    e, b, _ = train_to(eng, 0, 0, n_steps)
    return eng.state_hash(), (e, b)


def _resumed(engine_cls_a, engine_cls_b, corpus, tmp_path, cut: int, n_steps: int) -> str:
    cfg = EngineConfig(**CFG)
    a = engine_cls_a(cfg, corpus=corpus)
    e, b, _ = train_to(a, 0, 0, cut)
    path = tmp_path / "ckpt.npz"
    stored = save_checkpoint(path, a, e, b)
    assert stored == a.state_hash()
    cfg2, state, meta = load_checkpoint(path)
    assert cfg2 == cfg and meta["position"] == {"epoch": e, "batch_index": b} and meta["step"] == cut
    r = engine_cls_b(cfg2, corpus=corpus)
    r.load_state(state)
    assert r.state_hash() == stored
    train_to(r, meta["position"]["epoch"], meta["position"]["batch_index"], n_steps - cut)
    return r.state_hash()


@pytest.mark.parametrize("cut", [1, 7, 8, 33])
def test_cpu_resume_reproduces_the_uninterrupted_hash(tmp_path, cut):
    corpus = _corpus()
    full, _ = _uninterrupted(CpuEngine, corpus, 40)
    assert _resumed(CpuEngine, CpuEngine, corpus, tmp_path, cut, 40) == full


def test_periodic_checkpoints_and_next_position(tmp_path):
    corpus = _corpus()
    eng = CpuEngine(EngineConfig(**CFG), corpus=corpus)
    assert next_position(eng, 0, eng.n_batches() - 1) == (1, 0)
    path = tmp_path / "periodic.npz"
    e, b, totals = train_to(eng, 0, 0, 25, checkpoint_path=path, every=10)
    _, state, meta = load_checkpoint(path)
    assert meta["step"] == 25 and meta["position"] == {"epoch": e, "batch_index": b}
    assert totals["n_examples"] > 0
    assert np.array_equal(state["alive"], eng.state()["alive"])


def test_damaged_checkpoint_is_rejected(tmp_path):
    corpus = _corpus()
    eng = CpuEngine(EngineConfig(**CFG), corpus=corpus)
    train_to(eng, 0, 0, 3)
    path = tmp_path / "c.npz"
    save_checkpoint(path, eng, 0, 3)
    _, state, meta = load_checkpoint(path)
    state["alive"][0] ^= 1
    np.savez(path, __meta__=np.frombuffer(__import__("json").dumps(meta).encode(), dtype=np.uint8), **state)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def _cuda_engine_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    return CudaEngine


@pytest.mark.parametrize("cut", [5, 16])
def test_cuda_resume_reproduces_the_uninterrupted_hash_and_crosses_engines(tmp_path, cut):
    CudaEngine = _cuda_engine_or_skip()
    corpus = _corpus()
    full_cpu, _ = _uninterrupted(CpuEngine, corpus, 40)
    full_gpu, _ = _uninterrupted(CudaEngine, corpus, 40)
    assert full_cpu == full_gpu
    assert _resumed(CudaEngine, CudaEngine, corpus, tmp_path, cut, 40) == full_cpu
    assert _resumed(CpuEngine, CudaEngine, corpus, tmp_path, cut, 40) == full_cpu
    assert _resumed(CudaEngine, CpuEngine, corpus, tmp_path, cut, 40) == full_cpu
