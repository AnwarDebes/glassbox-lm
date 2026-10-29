"""docs/04 M4 ACCEPT: "GPU state hash == CPU hash after 100 steps at B in {64, 1024, 4096}; hash
invariant to launch configuration" (docs/13 H2.2, H2.3). Also equal DIAG_COUNTERS on every step,
prediction and fired-list equality, index-vs-brute-force agreement on the GPU, and state round
trips between the two engines.

Configs: the engine configs of tests/test_cpu_engine_parity.py (masked, causal, diffusion with
eager_exact decay and uniform_binomial negatives, dense-equivalent tabular), run on a larger toy
corpus (12,000 tokens) so that B = 1024 and B = 4096 are real batch sizes, plus one masked
V = 2000 config with 65,536 clauses. The dense-equivalent config runs at B = 1 only (docs/11 N9).

Skipped when CuPy or the GPU is absent. Run under the shared GPU lock:
    scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest tests/test_cuda_engine_parity.py
"""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import dense_config, noisy_xor, toy_corpus
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine

pytestmark = pytest.mark.gpu  # 2026-10-25 (TLM-B3): every test here needs the GPU
cp = pytest.importorskip("cupy")
try:
    cp.cuda.Device(0).compute_capability
except Exception as exc:  # pragma: no cover
    pytest.skip(f"no CUDA device: {exc}", allow_module_level=True)

from tmlm.engine.cuda.cuda_engine import CudaEngine  # noqa: E402

STEPS = 100
BATCHES = (64, 1024, 4096)


def _corpus40():
    return toy_corpus(vocab=40, n_docs=40, doc_len=300, seed=7)


def _corpus2000():
    return toy_corpus(vocab=2000, n_docs=40, doc_len=300, seed=11, n_stop=64)


def sparse_configs():
    yield "masked", dict(mode="masked", vocab=40, n_clauses=64, seed=1, sweep_period=8, n_bag_k=2), _corpus40
    yield "causal", dict(mode="causal", vocab=40, n_clauses=64, seed=3, sweep_period=16, n_bag_k=3, dead_sweeps=2), _corpus40
    yield "diffusion_eager", dict(mode="diffusion", vocab=40, n_clauses=48, seed=5, sweep_period=8, n_bag_k=2,
                                  decay_mode="eager_exact", neg_mode="uniform_binomial"), _corpus40
    yield "masked_v2000_c65536", dict(mode="masked", vocab=2000, n_clauses=65536, seed=1, sweep_period=32, dead_sweeps=2,
                                      n_bag_k=2), _corpus2000


def _run_steps(c: CpuEngine, g: CudaEngine, steps: int, name: str, hash_every: int = 25) -> tuple[int, int]:
    e = b = 0
    for s in range(steps):
        dc, dg = c.train_batch(e, b), g.train_batch(e, b)
        assert dc == dg, f"{name}: DIAG differs at step {s}: {dc} vs {dg}"
        if (s + 1) % hash_every == 0 or s + 1 == steps:
            assert c.state_hash() == g.state_hash(), f"{name}: state hash differs at step {s}"
        b += 1
        if b >= c.n_batches():
            e, b = e + 1, 0
    return e, b


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("name,over,corpus_fn", list(sparse_configs()), ids=[c[0] for c in sparse_configs()])
def test_cuda_engine_matches_cpu_engine_for_100_steps(name, over, corpus_fn, batch):
    corpus = corpus_fn()
    cfg = EngineConfig(batch=batch, **over)
    c, g = CpuEngine(cfg, corpus=corpus), CudaEngine(cfg, corpus=corpus)
    assert c.state_hash() == g.state_hash(), "initial state differs"
    epoch, _ = _run_steps(c, g, STEPS, f"{name} B={batch}")
    assert c.state_hash() == g.state_hash()
    ids = np.arange(0, c.n_examples, max(1, c.n_examples // 7))[:7]
    assert np.array_equal(c.predict_scores(epoch, ids), g.predict_scores(epoch, ids))
    for p in ids[:4]:
        assert np.array_equal(c.fired_clauses(epoch, int(p)), g.fired_clauses(epoch, int(p)))
        assert np.array_equal(g.fired_clauses(epoch, int(p)), g.fired_clauses_bruteforce(epoch, int(p)))
    assert c.n_free() == g.n_free()


def test_dense_equivalent_config_matches_at_batch_one():
    """The dense-equivalent tabular config of the CPU parity test (docs/11 section 6, B = 1 by N9)."""
    cfg = dense_config(12, 2, 24, 15, 3.9, seed=1)
    train = noisy_xor(400, 12, noise=0.1, seed=1)
    c, g = CpuEngine(cfg, tabular=train), CudaEngine(cfg, tabular=train)
    assert c.state_hash() == g.state_hash()
    epoch, _ = _run_steps(c, g, STEPS, "dense")
    ids = np.arange(0, 400, 57)[:7]
    assert np.array_equal(c.predict_scores(epoch, ids), g.predict_scores(epoch, ids))
    for p in ids[:4]:
        assert np.array_equal(c.fired_clauses(epoch, int(p)), g.fired_clauses(epoch, int(p)))
        assert np.array_equal(g.fired_clauses(epoch, int(p)), g.fired_clauses_bruteforce(epoch, int(p)))


def test_focused_negatives_config_matches():
    """neg_mode = focused (spec 5.3 top-K) exercises k6_select's ranking; not in the CPU parity set."""
    corpus = _corpus40()
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=2, batch=64, sweep_period=8, n_bag_k=2, neg_mode="focused")
    c, g = CpuEngine(cfg, corpus=corpus), CudaEngine(cfg, corpus=corpus)
    _run_steps(c, g, STEPS, "focused")
    assert c.state_hash() == g.state_hash()


def test_hash_is_invariant_to_the_launch_configuration():
    """docs/11 section 3: no draw may depend on the thread layout, grid or block sizes (H2.2)."""
    corpus = _corpus2000()
    cfg = EngineConfig(mode="masked", vocab=2000, n_clauses=65536, seed=1, batch=1024, sweep_period=16, dead_sweeps=2, n_bag_k=2)
    hashes = []
    diags = []
    for block, max_grid in ((64, 4096), (256, 4096), (128, 7)):
        g = CudaEngine(cfg, corpus=corpus, block=block, max_grid=max_grid)
        d = []
        e = b = 0
        for _ in range(40):
            d.append(g.train_batch(e, b))
            b += 1
            if b >= g.n_batches():
                e, b = e + 1, 0
        hashes.append(g.state_hash())
        diags.append(d)
    assert hashes[0] == hashes[1] == hashes[2]
    assert diags[0] == diags[1] == diags[2]


def test_state_round_trip_between_cpu_and_cuda_engines():
    corpus = _corpus40()
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=64, sweep_period=8, n_bag_k=2)
    c = CpuEngine(cfg, corpus=corpus)
    for b in range(min(20, c.n_batches())):
        c.train_batch(0, b)
    g = CudaEngine(cfg, corpus=corpus)
    g.load_state(c.state())
    assert g.state_hash() == c.state_hash()
    assert g.n_free() == c.n_free()
    for b in range(10):
        assert c.train_batch(1, b) == g.train_batch(1, b)
    assert c.state_hash() == g.state_hash()
    c2 = CpuEngine(cfg, corpus=corpus)
    c2.load_state(g.state())
    assert c2.state_hash() == g.state_hash()
    assert c2.train_batch(2, 0) == g.train_batch(2, 0)
    assert c2.state_hash() == g.state_hash()
