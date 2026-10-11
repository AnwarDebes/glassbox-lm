"""G6 deterministic replay (docs/07; docs/05 E11; docs/13 H2.2; docs/04 M8 ACCEPT "generated text
and logs byte-identical on replay"): the same seed, config and state give byte-identical
generated text and receipts across two runs and two thread counts (OMP_NUM_THREADS 1 and 2), and
training a small model twice gives the same state_hash. Everything runs in fresh subprocesses so
that no interpreter state can leak between the runs."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import sys, json, numpy as np
sys.path.insert(0, %(src)r)
from tmlm.data.synthetic import toy_corpus
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine
from tmlm.decode.scorer import TrainedModel
from tmlm.decode.conftm import fit_two_scalar
from tmlm.decode.anyorder import anyorder_generate
from tmlm.decode.diffusion import diffusion_generate, elbo_nll
from tmlm.decode.eliminate import elimination_generate
from tmlm.decode.incremental import IncrementalEvaluator
from tmlm.certify.receipt import build_receipts, write_receipts
corpus = toy_corpus(vocab=40, n_docs=6, doc_len=80, seed=7)
cfg = EngineConfig(mode="masked", vocab=40, n_clauses=256, seed=1, batch=32, sweep_period=8, n_bag_k=2)
eng = CpuEngine(cfg, corpus=corpus)
for e in range(3):
    for b in range(eng.n_batches()):
        eng.train_batch(e, b)
print("state_hash", eng.state_hash())
model = TrainedModel.from_engine(eng, corpus.tok)
pos = np.arange(0, corpus.tok.shape[0], 3, dtype=np.uint32)
cal, _ = fit_two_scalar(eng.predict_scores(0, pos), corpus.tok[pos], model.class_bias)
model = model.with_calibration(cal)
print("bank_sha256", model.bank_sha256(), "calibration", cal.a_q16, cal.b_q16)
prompt = corpus.tok[:8]
docs = []
for seed in (1, 2):
    for name, g in (("anyorder", anyorder_generate(model.bind(20), prompt, 12, seed=seed, order="confidence")),
                    ("diffusion", diffusion_generate(model.bind(20), prompt, 12, seed=seed, steps=4, remask_frac=0.2)),
                    ("eliminate", elimination_generate(model.bind(20), prompt, 12, seed=seed)),
                    ("anyorder_incremental", anyorder_generate(IncrementalEvaluator(model, 20), prompt, 12, seed=seed, order="random"))):
        print(name, seed, " ".join(str(int(t)) for t in g.generated))
        docs.append(build_receipts(g, model))
print("elbo", json.dumps(elbo_nll(model.bind(30), corpus.tok[:30], 5, seed=1), sort_keys=True))
write_receipts(%(out)r, {"runs": docs})
'''


def _run(n_threads: int, out: Path) -> tuple[bytes, bytes]:
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    code = SCRIPT % {"src": str(ROOT / "src"), "out": str(out)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=True)
    return r.stdout, out.read_bytes()


def test_generation_and_receipts_replay_byte_identically_across_runs_and_thread_counts(tmp_path):
    runs = [_run(1, tmp_path / "r1.json"), _run(2, tmp_path / "r2.json"), _run(1, tmp_path / "r3.json")]
    for stdout, receipts in runs[1:]:
        assert stdout == runs[0][0], "generated text differs between runs"
        assert receipts == runs[0][1], "receipts differ between runs"
    text = runs[0][0].decode()
    assert text.count("state_hash") == 1 and "anyorder 1 " in text and "eliminate 2 " in text
    # different seeds do not have to differ, but the text must carry both
    assert hashlib.sha256(runs[0][1]).hexdigest() == hashlib.sha256(runs[1][1]).hexdigest()


def test_training_twice_gives_the_same_state_hash():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tmlm.data.synthetic import toy_corpus\n"
        "from tmlm.engine.config import EngineConfig\n"
        "from tmlm.engine.cpu.cpu_engine import CpuEngine\n"
        "corpus = toy_corpus(vocab=40, n_docs=4, doc_len=60, seed=3)\n"
        "cfg = EngineConfig(mode='masked', vocab=40, n_clauses=128, seed=5, batch=16, sweep_period=8, n_bag_k=1)\n"
        "c = CpuEngine(cfg, corpus=corpus)\n"
        "[c.train_batch(e, b) for e in range(3) for b in range(c.n_batches())]\n"
        "print(c.state_hash())\n" % str(ROOT / "src")
    )
    outs = []
    for n in (1, 2):
        env = dict(os.environ, OMP_NUM_THREADS=str(n))
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip())
    assert len(set(outs)) == 1 and len(outs[0]) == 64
