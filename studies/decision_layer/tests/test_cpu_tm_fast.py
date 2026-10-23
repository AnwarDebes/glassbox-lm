"""The fast CPU decision (src/dl/cpu_tm_fast.py) against src/dl/cpu_tm.py: exact class sums.

    OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv-dl/bin/python tests/test_cpu_tm_fast.py

- random machines (empty clauses, plain and negated literals, with and without negation): class sums equal
  to dl.cpu_tm on random bits;
- view B (mem, words, xcust, xrecent; potion-32m) on dev, bits from src/dl/bitcache.py with the hashes of the
  R19 runs: STAR with e60_star_B_c16000's model, ABCD with R19abcd tm_mem_static seed 1's model. Class sums
  equal on every dev row, the dev accuracy as recorded, and decide() equal to the argmax on 500 rows.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import cpu_tm, cpu_tm_fast  # noqa: E402

sha = lambda X: hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()  # noqa: E731


def log(*a):
    print(*a, flush=True)


def test_random_machines():
    rng = np.random.default_rng(0)
    for negation in (True, False):
        for f, n_cl, n_cls in ((7, 50, 3), (64, 300, 5), (130, 1000, 11)):
            ta = rng.integers(1, 513, size=(n_cl, 2 * f if negation else f))
            ta[rng.random(ta.shape) < 0.9] = 1  # sparse inclusions
            ta[: n_cl // 10] = 1  # some empty clauses
            w = rng.integers(-50, 51, size=(n_cl, n_cls)).astype(np.int32)
            X = (rng.random((200, f)) < 0.5).astype(np.uint8)
            a = cpu_tm.class_sums(cpu_tm.pack(ta, w, negation=negation), X)
            m = cpu_tm_fast.pack(ta, w, negation=negation)
            b = cpu_tm_fast.class_sums(m, X)
            assert np.array_equal(a, b), (negation, f)
            assert all(cpu_tm_fast.decide(m, X[i]) == int(a[i].argmax()) for i in range(len(X)))
    log("random machines: class sums equal to cpu_tm (with and without negation, empty clauses)")


def test_view_b_dev():
    from dl.bitcache import view_bits

    cases = [("star", "results/explore/abcd_models/e60_star_B_c16000_seed1.npz", "results/confirm/R19star/tm_mem_static/seed1.json"),
             ("abcd", "results/confirm/R19abcd/tm_mem_static/model_seed1.npz", "results/confirm/R19abcd/tm_mem_static/seed1.json")]
    for ds, mpath, jpath in cases:
        Xtr, _ytr, Xdv, ydv, _names, _blist, _tr, _dv = view_bits(ds, ("mem", "words", "xcust", "xrecent"), encoder="potion-32m")
        j = json.loads((ROOT / jpath).read_text())
        assert sha(Xtr) == j["bits_sha256"]["train"] and sha(Xdv) == j["bits_sha256"]["dev"], ds
        z = np.load(ROOT / mpath)
        a = cpu_tm.class_sums(cpu_tm.pack(z["ta"], z["w"]), Xdv)
        m = cpu_tm_fast.pack(z["ta"], z["w"])
        b = cpu_tm_fast.class_sums(m, Xdv)
        assert np.array_equal(a, b), ds
        acc = float((b.argmax(1) == ydv).mean())
        assert acc == j["dev_acc"], (ds, acc, j["dev_acc"])
        assert all(cpu_tm_fast.decide(m, Xdv[i]) == int(a[i].argmax()) for i in range(500))
        log(f"view B {ds} dev: class sums equal on all {len(Xdv)} rows, dev accuracy {acc:.6f} as recorded, "
            f"{m['n_clauses_kept']} clauses kept, {m['n_literals']} literals")


if __name__ == "__main__":
    test_random_machines()
    test_view_b_dev()
    print("ok")
