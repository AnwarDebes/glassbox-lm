"""scripts/e87b_run.py on the CPU (dated note logs/plan_e87_note_2026-10-25.md; added 2026-10-25, before any Part B number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-dl/bin/python tests/test_e87b_run.py

CPU only; nothing is written in the study (the end-to-end part uses a temporary folder). It asserts:
- the reason for the wrapper: tm.py's own chunked update raises NotImplementedError for the exact budget;
- with the wrapper's patch and CHUNK_ELEMS forced small (60 clauses in 9 chunks), a machine with the exact budget 3,
  the trim start and the rank selection keeps every clause at 3 literals or fewer at the start and after every
  training step, runs the chunked budget update at every step (epochs x ceil(n / batch size) calls), carries the
  wrapper's neg_sampling in its config, and learns a small task (dev accuracy above 0.8);
- the same machine unchunked and unpatched (tm.py's own exact budget) also keeps the budget, as a reference;
- refusals at the machine: no budget (max_literals 0), budget_mode "tmu", a negated cap;
- select_k_skip0 equals dl.tm.select_k on 400 random cases (ties among the keys, rows with k = 0, k above the number
  of crossings, rows with no crossing; "rank" and "threshold"), and two chunked trainings, one with tm_chunkbudget's
  select_k left as dl.tm.select_k (--plain-select) and one with select_k_skip0, end with the same ta, w and history;
- end to end, with a made-up target script that builds the machine on the CPU and saves its model file the way
  abcd_explore.py --save does: the run-info file (knobs, CHUNK_ELEMS, the step count, the model file's SHA-256, the
  select used: skip0 by default, plain with --plain-select); refusals: an existing run-info file or model file (the
  target does not start), no --save, a target that fails after saving its model (no run-info), a tmu budget, no --
  separator, an unknown negative rule.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import dl.tm as tm_mod  # noqa: E402
import dl.tm_chunkbudget as cb  # noqa: E402
import e87b_run as W  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None  # CoTM.fit ends with it; this test runs on the CPU

TARGET = '''
import argparse, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl.tm import TMConfig
from dl.tm_init import make_tm
torch.cuda.synchronize = lambda *a, **k: None
ap = argparse.ArgumentParser()
ap.add_argument("--tag"); ap.add_argument("--seed", type=int); ap.add_argument("--save", action="store_true")
ap.add_argument("--budget-mode", default="exact"); ap.add_argument("--fail", action="store_true")
a = ap.parse_args()
rng = np.random.default_rng(0)
X = (rng.uniform(size=(400, 10)) < 0.3).astype(np.uint8)
y = (X[:, 0] & X[:, 1]).astype(np.int64)
cfg = TMConfig(n_clauses=20, T=20, s=4.0, epochs=2, max_literals=3, budget_mode=a.budget_mode, budget_select="rank",
               seed=a.seed, batch_size=64)
tm = make_tm(2, 10, cfg, init="trim", device="cpu")
tm.fit(X[:300], y[:300], X[300:], y[300:])
if a.save:
    sd = ROOT / "results" / "explore" / "abcd_models"
    sd.mkdir(parents=True, exist_ok=True)
    st = tm.state()
    np.savez_compressed(sd / f"{a.tag}_seed{a.seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"])
if a.fail:
    raise RuntimeError("made-up failure after the model was saved")
print("fake target done", flush=True)
'''


def task(n=3000, f=40, seed=0):
    """tests/test_tm_chunkbudget.py's task: three classes from two short rules."""
    rng = np.random.default_rng(seed)
    X = (rng.uniform(size=(n, f)) < 0.3).astype(np.uint8)
    y = np.minimum((X[:, 0] & X[:, 1]) + 2 * (X[:, 2] & ~X[:, 3] & 1), 2).astype(np.int64)
    return X, y


def cfg(**kw):
    base = dict(n_clauses=60, T=60, s=4.0, epochs=6, max_literals=3, budget_select="rank", seed=3, batch_size=64)
    return TMConfig(**{**base, **kw})


def watch(tm, out: list) -> None:
    """Record the longest clause after every training step of this machine."""
    inner = tm.feedback

    def fb(*a, **k):
        inner(*a, **k)
        out.append(int((tm.ta > tm.cfg.n_states).sum(1).max()))

    tm.feedback = fb


def select_cases() -> None:
    """The rows select_k gets in training are the "over" rows: at least one crossing, k (the allowance) below the number
    of crossings. Both modes on that domain; "rank" also with k up to the row width and rows with no crossing."""
    rng = np.random.default_rng(5)
    n = 0
    for case in range(400):
        r, m = int(rng.integers(1, 9)), int(rng.integers(1, 40))
        cr = torch.as_tensor(rng.random((r, m)) < rng.uniform(0.05, 0.9))
        vals = rng.integers(0, 4, (r, m)) / 4.0 if case % 2 else rng.random((r, m))  # every other case: many ties
        keys = torch.as_tensor(vals, dtype=torch.float32).masked_fill(~cr, -1.0)
        nc = cr.sum(1).numpy()
        zero = rng.random(r) < 0.4  # about 40% of the rows keep nothing (a clause at the budget)
        over = torch.as_tensor(nc >= 1)
        k = torch.as_tensor(np.where(zero, 0, rng.integers(0, np.maximum(nc, 1))))
        if bool(over.any()):
            for mode in ("rank", "threshold"):
                assert torch.equal(W.select_k_skip0(keys[over], cr[over], k[over], mode), tm_mod.select_k(keys[over], cr[over], k[over], mode)), (case, mode)
                n += 1
        kw = torch.as_tensor(np.where(zero, 0, rng.integers(0, m + 1, r)))
        assert torch.equal(W.select_k_skip0(keys, cr, kw, "rank"), tm_mod.select_k(keys, cr, kw, "rank")), case
    print(f"ok select_k_skip0 equals dl.tm.select_k: {n} rank and threshold calls on over rows (ties, k = 0 rows), and 400 rank calls"
          " with k up to the row width and rows with no crossing")


def trained(plain: bool, f: int, X, y, old_init):
    tm_mod.CoTM.__init__ = old_init  # a fresh patch each time
    state = {"machines": []}
    W.install("hardest", 2 * f * 7, state, plain=plain)
    inner, rows = cb.select_k, {"k0": 0, "kpos": 0}

    def counting(keys, cr, k, mode="threshold"):
        rows["k0"] += int((k == 0).sum())
        rows["kpos"] += int((k > 0).sum())
        return inner(keys, cr, k, mode)

    cb.select_k = counting
    tm = make_tm(3, f, cfg(epochs=4), init="trim", device="cpu")
    tm.fit(X[:2400], y[:2400], X[2400:], y[2400:])
    cb.select_k = inner
    return tm, {**state["machines"][0], **rows}


def in_process() -> None:
    X, y = task()
    f = X.shape[1]
    old_elems, old_init, old_sel = tm_mod.CHUNK_ELEMS, tm_mod.CoTM.__init__, cb.select_k
    try:
        # the reason: tm.py's own chunked update refuses the exact budget
        tm_mod.CHUNK_ELEMS = 2 * f * 7
        tm = make_tm(3, f, cfg(), init="trim", device="cpu")
        try:
            tm.fit(X[:256], y[:256])
            raise AssertionError("tm.py's chunked update must raise for the exact budget")
        except NotImplementedError:
            pass
        tm_mod.CHUNK_ELEMS = old_elems
        print("ok tm.py's own chunked update raises NotImplementedError for the exact budget (the reason for the wrapper)")

        state = {"machines": []}
        W.install("random", 2 * f * 7, state)  # 7 clauses per chunk: 60 clauses in 9 chunks
        tm = make_tm(3, f, cfg(), init="trim", device="cpu")
        assert int((tm.ta > tm.cfg.n_states).sum(1).max()) <= 3, "the trim start obeys the budget"
        lens: list = []
        watch(tm, lens)
        tm.fit(X[:2400], y[:2400], X[2400:], y[2400:])
        acc = float((tm.predict(torch.as_tensor(X[2400:], dtype=torch.uint8)).numpy() == y[2400:]).mean())
        steps = 6 * math.ceil(2400 / 64)
        m = state["machines"][0]
        assert len(lens) == steps and max(lens) <= 3, (len(lens), max(lens))
        assert m["chunked"] and m["clauses_per_chunk"] == 7 and m["chunked_budget_steps"] == steps, m
        assert tm.cfg.neg_sampling == "random" and tm.stats()["max_literals"] <= 3 and acc > 0.8, (tm.cfg.neg_sampling, acc)
        print(f"ok wrapper patch: exact budget 3, trim start, 60 clauses in 9 chunks: longest clause {max(lens)} over {steps} steps, "
              f"{m['chunked_budget_steps']} chunked budget updates, neg_sampling random, dev accuracy {acc:.3f}")

        for bad, why in ((dict(max_literals=0), "max_literals 0"), (dict(budget_mode="tmu"), "budget_mode 'tmu'"),
                         (dict(max_negated=2), "max_negated 2")):
            try:
                tm_mod.CoTM(3, f, cfg(**bad), device="cpu")
                raise AssertionError(f"{bad} must be refused")
            except SystemExit as e:
                assert "needs an exact literal budget" in str(e) and why in str(e), str(e)
        print("ok refusals at the machine: no budget, budget_mode tmu, a negated cap")

        a, ma = trained(True, f, X, y, old_init)
        assert cb.select_k is tm_mod.select_k
        b, mb = trained(False, f, X, y, old_init)
        assert cb.select_k is W.select_k_skip0 and ma["chunked_budget_steps"] == mb["chunked_budget_steps"] == 4 * math.ceil(2400 / 64)
        h = lambda t: [(e["epoch"], e["val_acc"]) for e in t.history]  # noqa: E731 (the field "t" is wall-clock time)
        assert torch.equal(a.ta, b.ta) and torch.equal(a.w, b.w) and h(a) == h(b) and a.best_epoch == b.best_epoch, (h(a), h(b))
        assert mb["k0"] > 0 and mb["kpos"] > 0 and (ma["k0"], ma["kpos"]) == (mb["k0"], mb["kpos"]), (ma, mb)
        print(f"ok skip0 against plain select: two chunked trainings (4 epochs, {ma['chunked_budget_steps']} steps) end with the same ta, w and"
              f" history, bit for bit; over rows with k = 0 (not sorted by skip0): {mb['k0']}, with k > 0: {mb['kpos']}")
    finally:
        tm_mod.CHUNK_ELEMS, tm_mod.CoTM.__init__, cb.select_k = old_elems, old_init, old_sel

    tm = make_tm(3, f, cfg(), init="trim", device="cpu")  # unpatched, unchunked: tm.py's own exact budget
    lens = []
    watch(tm, lens)
    tm.fit(X[:2400], y[:2400], X[2400:], y[2400:])
    acc_u = float((tm.predict(torch.as_tensor(X[2400:], dtype=torch.uint8)).numpy() == y[2400:]).mean())
    assert max(lens) <= 3 and acc_u > 0.8, (max(lens), acc_u)
    print(f"ok reference: unchunked, unpatched (tm.py's exact budget): longest clause {max(lens)}, dev accuracy {acc_u:.3f}")


def end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    try:
        (tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "e87b_run.py", tmp / "scripts" / "e87b_run.py")
        (tmp / "src").symlink_to(ROOT / "src")
        (tmp / "scripts" / "fake_target.py").write_text(TARGET)
        info = tmp / "results/explore/e87/b_runs/fk_seed2.json"
        model = tmp / "results/explore/abcd_models/fk_seed2.npz"

        def run(*wrap, target=("--save", "--tag", "fk", "--seed", "2")):
            cmd = [sys.executable, "scripts/e87b_run.py", *wrap, "--", "scripts/fake_target.py", *target]
            return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env=env)

        good = ("--neg-sampling", "random", "--chunk-elems", "100", "--run-info", "results/explore/e87/b_runs/fk_seed2.json")
        p = run(*good)
        assert p.returncode == 0 and "fake target done" in p.stdout, p.stdout + p.stderr
        rec = json.loads(info.read_text())
        m = rec["machines"]
        assert rec["knobs"] == {"neg_sampling": "random", "chunk_budget": "dl.tm_chunkbudget.enable", "select": "skip0"} and rec["chunk_elems"] == 100, rec
        assert len(m) == 1 and m[0]["chunked"] and m[0]["clauses_per_chunk"] == 5 and m[0]["chunked_budget_steps"] == 2 * math.ceil(300 / 64), m
        assert rec["model"] == "results/explore/abcd_models/fk_seed2.npz" and rec["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest(), rec
        assert rec["argv"] == ["scripts/fake_target.py", "--save", "--tag", "fk", "--seed", "2"], rec
        assert json.loads(str(np.load(model)["cfg"]))["neg_sampling"] == "random"
        print("ok end to end: run-info with the knobs, CHUNK_ELEMS 100, 10 chunked budget updates (2 epochs x 5 steps), the model's SHA-256")

        p = run(*good)
        assert p.returncode == 1 and "exists; not overwriting" in p.stderr and "fake target done" not in p.stdout, p.stdout + p.stderr
        info.unlink()
        p = run(*good)
        assert p.returncode == 1 and "fk_seed2.npz exists; not overwriting" in p.stderr and not info.exists(), p.stdout + p.stderr
        model.unlink()
        p = run(*good, target=("--tag", "fk", "--seed", "2"))
        assert p.returncode == 1 and "needs --save, --tag and --seed" in p.stderr, p.stdout + p.stderr
        p = run(*good, target=("--save", "--tag", "fk", "--seed", "2", "--fail"))
        assert p.returncode != 0 and "made-up failure" in p.stderr and model.exists() and not info.exists(), p.stdout + p.stderr
        model.unlink()
        p = run(*good, target=("--save", "--tag", "fk", "--seed", "2", "--budget-mode", "tmu"))
        assert p.returncode == 1 and "needs an exact literal budget" in p.stderr and not info.exists() and not model.exists(), p.stdout + p.stderr
        p = subprocess.run([sys.executable, "scripts/e87b_run.py", *good, "scripts/fake_target.py"], capture_output=True, text=True, cwd=tmp, env=env)
        assert p.returncode == 1 and "put -- before the script" in p.stderr, p.stdout + p.stderr
        p = run("--neg-sampling", "mix:0.5", "--run-info", "x.json")
        assert p.returncode == 2 and "invalid choice" in p.stderr, p.stdout + p.stderr
        print("ok refusals: an existing run-info file, an existing model file, no --save, a failing target (no run-info), a tmu budget,"
              " no -- separator, an unknown negative rule")
        p = run("--neg-sampling", "hardest", "--plain-select", "--chunk-elems", "100", "--run-info", "results/explore/e87/b_runs/fk_seed2.json")
        assert p.returncode == 0 and json.loads(info.read_text())["knobs"]["select"] == "plain", p.stdout + p.stderr
        print("ok --plain-select is recorded as select plain")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    select_cases()
    in_process()
    end_to_end()
    print("PASSED")


if __name__ == "__main__":
    main()
