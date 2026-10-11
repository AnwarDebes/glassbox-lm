"""docs/04 M3 ACCEPT: "oracle and C++ state hashes equal for 200 steps on 3 configs". Also equal
DIAG_COUNTERS on every step, index-vs-brute-force agreement, prediction equality, state round
trips, and invariance to the OpenMP thread count (docs/11 section 3: no draw may depend on the
thread layout)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.data.synthetic import dense_config, noisy_xor, toy_corpus
from tmlm.engine.api import DIAG_COUNTERS
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine
from tmlm.engine.oracle.oracle import OracleEngine

ROOT = Path(__file__).resolve().parents[1]


def configs():
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    yield "masked", EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2), corpus, None
    yield "causal", EngineConfig(mode="causal", vocab=40, n_clauses=64, seed=3, batch=8, sweep_period=16, n_bag_k=3, dead_sweeps=2), corpus, None
    yield "diffusion_eager", EngineConfig(mode="diffusion", vocab=40, n_clauses=48, seed=5, batch=16, sweep_period=8, n_bag_k=2, decay_mode="eager_exact", neg_mode="uniform_binomial"), corpus, None
    train = noisy_xor(400, 12, noise=0.1, seed=1)
    yield "dense", dense_config(12, 2, 24, 15, 3.9, seed=1), None, train


@pytest.mark.parametrize("name,cfg,corpus,tabular", list(configs()), ids=[c[0] for c in configs()])
def test_cpu_engine_is_bit_exact_with_the_oracle_for_200_steps(name, cfg, corpus, tabular):
    o = OracleEngine(cfg, corpus=corpus, tabular=tabular)
    c = CpuEngine(cfg, corpus=corpus, tabular=tabular)
    assert o.state_hash() == c.state_hash(), "initial state differs"
    steps = 0
    epoch = 0
    while steps < 200:
        for b in range(o.n_batches()):
            do, dc = o.train_batch(epoch, b), c.train_batch(epoch, b)
            assert do == dc, f"{name}: DIAG differs at step {steps}: {do} vs {dc}"
            assert o.state_hash() == c.state_hash(), f"{name}: state hash differs at step {steps}"
            steps += 1
            if steps >= 200:
                break
        epoch += 1
    assert o.check_invariants() == []
    # prediction and fired lists agree, and the C index agrees with brute force
    ids = np.arange(0, o.n_examples, max(1, o.n_examples // 7))[:7]
    assert np.array_equal(o.predict_scores(epoch, ids), c.predict_scores(epoch, ids))
    for p in ids[:4]:
        assert np.array_equal(o.fired_clauses(epoch, int(p)), c.fired_clauses(epoch, int(p)))
        assert np.array_equal(c.fired_clauses(epoch, int(p)), c.fired_clauses_bruteforce(epoch, int(p)))


def test_state_round_trip_between_the_two_engines():
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2)
    c = CpuEngine(cfg, corpus=corpus)
    for b in range(c.n_batches()):
        c.train_batch(0, b)
    o = OracleEngine(cfg, corpus=corpus)
    o.load_state(c.state())
    assert o.state_hash() == c.state_hash()
    assert o.check_invariants() == []
    o.train_batch(1, 0)
    c.train_batch(1, 0)
    assert o.state_hash() == c.state_hash()


def _hash_with_threads(n_threads: int) -> str:
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tmlm.data.synthetic import toy_corpus\n"
        "from tmlm.engine.config import EngineConfig\n"
        "from tmlm.engine.cpu.cpu_engine import CpuEngine\n"
        "corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)\n"
        "cfg = EngineConfig(mode='masked', vocab=40, n_clauses=64, seed=1, batch=32, sweep_period=8, n_bag_k=2)\n"
        "c = CpuEngine(cfg, corpus=corpus)\n"
        "[c.train_batch(e, b) for e in range(3) for b in range(c.n_batches())]\n"
        "print(c.state_hash())\n" % str(ROOT / "src")
    )
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_hash_does_not_depend_on_the_thread_count():
    assert _hash_with_threads(1) == _hash_with_threads(4)


def test_diag_counter_names_match():
    assert len(DIAG_COUNTERS) == 16
