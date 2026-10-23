"""scripts/e87_readability.py on hand-made models (plan logs/plan_e87_receipts.md; added 2026-10-23, before any e87 number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e87_readability.py

CPU only; nothing written in the study. It asserts:
- fires: plain and negated literals, an empty clause that never fires;
- TM receipts: one clause enough; two needed (the strongest clause alone favours a third class); ties by the lower
  clause index; no receipt within the cap;
- LR receipts: the intercept alone (0 bits); two bits needed;
- coverage at k literals and S1's point for the TM (3 clauses of 3 literals) and LR (9 bits);
- Part B's kill rule: the most accurate C at the budget, killed at exactly -0.5; the extension (0.003, 0.001, 0.0003,
  0.0001) stops at the first C that reaches 0.90 or once accuracy is below the pruned agent minus 5; no comparator
  gives "not testable";
- clause precision, the 10-firing floor, and the dev accuracy of a tiny model;
- end to end on a made-up A0 file and model: the output file, the dev-accuracy check against the run's seed json,
  a refusal when they differ, and a refusal to overwrite.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e87_readability as R  # noqa: E402

NS = 256


def ta_from(incl: np.ndarray) -> np.ndarray:
    return np.where(incl, NS + 1, 1).astype(np.int16)


def main() -> None:
    F = 3
    incl = np.zeros((4, 2 * F), dtype=bool)
    incl[0, 0] = True                      # x0
    incl[1, F + 1] = True                  # NOT x1
    incl[3, 0] = incl[3, F + 2] = True     # x0 AND NOT x2; clause 2 is empty
    X = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    got = R.fires(X, incl)
    want = np.array([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool)
    assert np.array_equal(got, want), got
    print("ok fires: plain and negated literals; an empty clause never fires")

    lens = np.array([1, 2, 3, 4])
    w = np.array([[5, 0, 0], [0, 4, 0], [3, 0, 2], [1, 1, 0]])
    S = w.sum(0, keepdims=True)
    c, r = R.decide(S)
    assert (c[0], r[0]) == (0, 1)
    D = (w[:, c].T - w[:, r].T)
    out = R.receipts(D, w, None, c, lens)
    assert out["n_items"][0] == 1 and out["literals"][0] == 1 and out["longest"][0] == 1, out
    w = np.array([[3, 0, 4], [2, 0, 0], [0, 4, 0]])
    S = w.sum(0, keepdims=True)
    c, r = R.decide(S)
    assert (c[0], r[0]) == (0, 1), (c, r)  # classes 1 and 2 tie at 4: the runner-up is the lower index
    D = (w[:, c].T - w[:, r].T)
    out = R.receipts(D, w, None, c, np.array([2, 3, 1]))
    assert out["n_items"][0] == 2 and out["literals"][0] == 5 and out["longest"][0] == 3, out
    short = R.receipts(D, w, None, c, np.array([2, 3, 1]), kmax=1)
    assert short["n_items"][0] == 2 and not short["found"][0] and short["literals"][0] == -1, short
    w = np.array([[2, 0], [2, 0], [0, 1]])
    D = np.array([[2, 2, -1]])
    out = R.receipts(D, w, None, np.array([0]), np.array([7, 1, 1]))
    assert out["n_items"][0] == 1 and out["literals"][0] == 7, out  # the tie goes to clause 0 (7 literals), not clause 1
    print("ok TM receipts: one clause; two needed; ties by the lower index; none within the cap")

    W, b = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0])
    x = np.array([[0, 0, 0]], dtype=np.uint8)
    c, r = R.decide(x @ W + b)
    out = R.receipts((W[:, c].T - W[:, r].T) * x, W, b, c, np.ones(3, dtype=np.int64))
    assert out["n_items"][0] == 0 and out["literals"][0] == 0 and out["found"][0], out
    b = np.array([0.0, 1.5])
    x = np.array([[1, 1, 0]], dtype=np.uint8)
    c, r = R.decide(x @ W + b)
    out = R.receipts((W[:, c].T - W[:, r].T) * x, W, b, c, np.ones(3, dtype=np.int64))
    assert c[0] == 0 and out["n_items"][0] == 2 and out["literals"][0] == 2, out
    print("ok LR receipts: the intercept alone (0 bits); two bits needed")

    rec = {"n_items": np.array([1, 2, 4, 21]), "literals": np.array([2, 9, 4, -1]), "longest": np.array([2, 5, 1, -1]),
           "found": np.array([True, True, True, False])}
    s = R.summary(rec, tm=True)
    cov = s["coverage_at_k_literals"]
    assert (cov["1"], cov["2"], cov["4"], cov["9"]) == (0.0, 0.25, 0.5, 0.75) and s["s1_point"] == 0.25, s
    assert s["share_no_receipt_within_20"] == 0.25 and R.summary(rec, tm=False)["s1_point"] == 0.75
    print("ok coverage at k literals; S1's point for the TM (3 clauses of 3 literals) and LR (9 bits)")

    grid = {0.01: {"dev_acc": 0.30, "cov9": 0.95}, 0.03: {"dev_acc": 0.33, "cov9": 0.91}, 0.1: {"dev_acc": 0.36, "cov9": 0.80}}
    v = R.l1_at_budget(dict(grid), 0.335)
    assert (v["C"], v["verdict"], v["added"]) == (0.03, "killed", []), v  # 0.33 >= 0.335 - 0.005: killed at exactly 0.5
    assert R.l1_at_budget(dict(grid), 0.3351)["verdict"] == "not killed"
    dense = {C: {"dev_acc": 0.36, "cov9": 0.5} for C in (0.01, 0.03, 0.1, 0.3, 1.0)}
    calls = []

    def fit(C, table={0.003: (0.35, 0.7), 0.001: (0.34, 0.92), 0.0003: (0.30, 0.99), 0.0001: (0.2, 1.0)}):
        calls.append(C)
        return {"dev_acc": table[C][0], "cov9": table[C][1]}
    v = R.l1_at_budget(dict(dense), 0.345, fit)
    assert (v["C"], v["verdict"], calls) == (0.001, "killed", [0.003, 0.001]), (v, calls)  # stops at the first that reaches 0.90
    calls.clear()
    v = R.l1_at_budget(dict(dense), 0.41, fit)
    assert (v["C"], v["verdict"], calls) == (None, "not testable", [0.003]), (v, calls)  # 0.35 < 0.41 - 0.05: stop
    v = R.l1_at_budget(dict(dense), 0.345, lambda C: {"dev_acc": 0.33, "cov9": 0.6})
    assert v["verdict"] == "not testable" and v["added"] == list(R.L1_EXT) and v["why"] == "L1-LR does not reach the budget", v
    print("ok Part B's kill rule: the most accurate C at the budget; killed at exactly -0.5; the extension 0.003 > 0.001 > ... stops at"
          " the first that reaches 0.90 or below the pruned agent minus 5; no comparator is 'not testable'")

    n = 20
    Xd = np.zeros((n, 2), dtype=np.uint8)
    Xd[:12, 0] = 1
    y = np.array([0] * 9 + [1] * 3 + [1] * 8)
    inc = np.zeros((2, 4), dtype=bool)
    inc[0, 0] = True        # x0, votes class 0
    inc[1, 2] = True        # NOT x0, votes class 1 (fires on 8 rows: below the 10-firing floor)
    t = R.tm_part(Xd, y, ta_from(inc), np.array([[3, 0], [0, 2]]), NS)
    assert t["dev_acc"] == 0.85 and t["precision"]["n_clauses_ge10"] == 1 and t["precision"]["median"] == 0.75, t
    assert t["clauses"]["non_empty"] == 2 and t["clauses"]["share_negated_only"] == 0.5 and t["firing"]["median_per_decision"] == 1.0, t
    print("ok clause precision with the 10-firing floor; dev accuracy; clause statistics")

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "e87_readability.py", tmp / "scripts" / "e87_readability.py")
        rng = np.random.default_rng(0)
        Xtr = rng.integers(0, 2, (60, 4)).astype(np.uint8)
        ytr = (Xtr[:, 0] + Xtr[:, 1]).astype(np.int64)  # 3 classes
        Xdv = rng.integers(0, 2, (20, 4)).astype(np.uint8)
        ydv = (Xdv[:, 0] + Xdv[:, 1]).astype(np.int64)
        h = {"train": hashlib.sha256(Xtr.tobytes()).hexdigest(), "dev": hashlib.sha256(Xdv.tobytes()).hexdigest()}
        Wl, bl = np.eye(4, 3).astype(np.float32), np.zeros(3, dtype=np.float32)
        (tmp / "results/explore/e87").mkdir(parents=True)
        np.savez_compressed(tmp / "results/explore/e87/abcd_c4.npz", X_train=Xtr, y_train=ytr, X_dev=Xdv, y_dev=ydv,
                            names=np.array(["a", "b", "c", "d"]), buttons=np.array(["k0", "k1", "k2"]), lr_W=Wl, lr_b=bl,
                            lr_C=np.float64(1.0), lr_pred_dev=(Xdv @ Wl + bl).argmax(1), bits_sha256=json.dumps(h), table="{}")
        mdir = tmp / R.MODELS["abcd"]
        mdir.mkdir(parents=True)
        inc = np.zeros((3, 8), dtype=bool)
        inc[0, 4], inc[0, 5] = True, True              # NOT x0 AND NOT x1 -> class 0
        inc[1, 0], inc[1, 1] = True, True              # x0 AND x1 -> class 2
        inc[2, 0] = True                               # x0 -> class 1
        wm = np.array([[5, 0, 0], [0, 0, 9], [0, 4, 0]], dtype=np.int32)
        np.savez_compressed(mdir / "model_seed1.npz", ta=ta_from(inc), w=wm, cfg=json.dumps({"n_states": NS}))
        acc = R.tm_part(Xdv, ydv, ta_from(inc), wm, NS)["dev_acc"]
        (mdir / "seed1.json").write_text(json.dumps({"dev_acc": acc, "bits_sha256": h}))
        cmd = [sys.executable, str(tmp / "scripts" / "e87_readability.py"), "--dataset", "abcd", "--seeds", "1", "--l1-grid", "1.0"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        assert p.returncode == 0, p.stdout + p.stderr
        res = json.loads((tmp / "results/explore/e87/readability_abcd.json").read_text())
        assert res["tm"]["1"]["dev_acc"] == acc and res["lr"]["agrees_with_A0_gpu_predictions"] == 1.0 and "1" in res["l1"], res
        assert set(res["tm_mean"]["coverage_at_k_literals"]) == {str(k) for k in range(1, 11)} and res["l1_chosen_on_dev"] == "1", res
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        assert p.returncode == 1 and "not overwriting" in p.stderr, p.stdout + p.stderr
        (tmp / "results/explore/e87/readability_abcd.json").unlink()
        (mdir / "seed1.json").write_text(json.dumps({"dev_acc": acc + 0.05, "bits_sha256": h}))
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        assert p.returncode == 1 and "is not the run's" in p.stderr, p.stdout + p.stderr
    finally:
        shutil.rmtree(tmp)
    print("ok end to end: output file, the dev-accuracy check against the run, a refusal when it differs, no overwrite")
    print("PASSED")


if __name__ == "__main__":
    main()
