"""The CPU decision path (src/dl/cpu_tm.py) gives exactly the class sums of CoTM, and reproduces a saved model.

    OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_cpu_tm.py

- On R18's TM-mem model of seed 1 and the cached STAR dev bits: cpu_tm.class_sums equals
  CoTM(device="cpu").class_sums on 300 dev rows, and its argmax over all dev rows reproduces the
  model's registered dev accuracy, 0.6394035900405327.
- On a small random machine with negated literals and empty clauses: equal sums.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import cpu_tm  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def cotm_cpu(ta, w, cfg_json, n_classes, n_bits):
    tm = CoTM(n_classes, n_bits, TMConfig(**json.loads(cfg_json)), device="cpu")
    tm.ta = torch.as_tensor(ta.astype(np.int32))
    tm.w = torch.as_tensor(w.astype(np.int32))
    return tm


def test_random_machine():
    rng = np.random.default_rng(0)
    N, f, C = 50, 70, 4
    ta = rng.integers(1, 513, size=(N, 2 * f))
    ta[rng.uniform(size=ta.shape) < 0.9] = 100  # sparse clauses
    ta[:5] = 100  # empty clauses
    w = rng.integers(-20, 21, size=(N, C))
    X = (rng.uniform(size=(500, f)) < 0.5).astype(np.uint8)
    tm = cotm_cpu(ta, w, json.dumps({"n_clauses": N, "T": N, "half": False}), C, f)
    ref = tm.class_sums(torch.as_tensor(X)).numpy().astype(np.int64)
    got = cpu_tm.class_sums(cpu_tm.pack(ta, w), X)
    assert np.array_equal(ref, got)
    print("random machine: cpu_tm == CoTM on 500 inputs", flush=True)


def test_r18_model():
    z = np.load(ROOT / "results" / "confirm" / "R18" / "tm_mem" / "model_seed1.npz")
    dv = np.load(ROOT / "data" / "star" / "bits_mem-intent-step-words_dev.npz")
    names = json.loads((ROOT / "results" / "confirm" / "R18" / "tm_mem" / "bit_names.json").read_text())
    assert json.loads(str(dv["names"])) == names
    X, y = dv["X"], dv["y"]
    model = cpu_tm.pack(z["ta"], z["w"])
    S = cpu_tm.class_sums(model, X)
    acc = float((S.argmax(1) == y).mean())
    rec = json.loads((ROOT / "results" / "confirm" / "R18" / "tm_mem" / "seed1.json").read_text())
    assert acc == rec["dev_acc"] == 0.6394035900405327, (acc, rec["dev_acc"])
    tm = cotm_cpu(z["ta"], z["w"], str(z["cfg"]), z["w"].shape[1], X.shape[1])
    ref = tm.class_sums(torch.as_tensor(X[:300])).numpy().astype(np.int64)
    assert np.array_equal(ref, S[:300])
    print(f"R18 tm_mem seed 1: {model['n_clauses_kept']} of {model['n_clauses']} clauses kept; CPU dev accuracy {acc} "
          f"(registered {rec['dev_acc']}); sums equal CoTM on 300 rows", flush=True)


if __name__ == "__main__":
    test_random_machine()
    test_r18_model()
    print("ok")
