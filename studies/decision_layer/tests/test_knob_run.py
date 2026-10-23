"""scripts/knob_run.py on the CPU (plan logs/plan_e83_learning_dynamics.md; added 2026-10-21).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_knob_run.py

CPU only; no GPU, no data read, nothing written in the study. A stub target in a temporary folder trains a tiny CoTM
(12 clauses, 6 features, 3 classes, 2 epochs, an evaluation every epoch) on made-up rows, on the CPU. It prints its
cfg, its final dev predictions, its evaluation accuracies and the sha256 of its automata and weights. The stub sets
torch.cuda.synchronize to a no-op, because tm.fit ends with a CUDA sync. It asserts:
- refusals: no "--", no script, a bad --neg-sampling (hard, mix:1.5, mix:0, mix:x), a bad --clause-drop (1.0, -0.1,
  abc), and an existing --preds-out (the target does not run);
- knob-off identity: the stub through knob_run.py with no knob, with and without --preds-out, prints exactly what the
  plain stub prints;
- the preds file: 3 predictions (2 evaluations, then the final call), the last equal to the stub's final
  predictions, y_val equal to the stub's dev labels, and each evaluation's accuracy equal to its history entry;
- knobs on: neg_sampling random and clause_drop 0.5 reach the cfg and change the trained machine; mix:0.5 reaches the
  cfg;
- a target that exits 3: knob_run.py exits 3 and writes no preds file.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STUB = r'''
import hashlib, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
S = Path(sys.argv[1])
sys.path.insert(0, str(S / "src"))
from dl.tm import CoTM, TMConfig
if len(sys.argv) > 3 and sys.argv[3] == "fail":
    sys.exit(3)
torch.cuda.synchronize = lambda *a, **k: None  # the CPU test: tm.fit ends with a CUDA sync
cfg = TMConfig(n_clauses=12, T=12, s=3.0, epochs=2, eval_every=1, batch_size=8, seed=int(sys.argv[2]))
rng = np.random.default_rng(0)
X = (rng.random((40, 6)) < 0.4).astype(np.uint8); y = rng.integers(0, 3, 40)
Xv = (rng.random((15, 6)) < 0.4).astype(np.uint8); yv = rng.integers(0, 3, 15)
tm = CoTM(3, 6, cfg, device="cpu").fit(X, y, Xv, yv)
pred = tm.predict(torch.as_tensor(Xv, dtype=torch.uint8)).numpy()
h = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
print("STUB " + json.dumps({"cfg": asdict(cfg), "pred": pred.tolist(), "yv": yv.tolist(), "val": [e["val_acc"] for e in tm.history],
                            "ta": h(tm.ta), "w": h(tm.w)}))
'''


def run(tmp: Path, *knob_args: str, fail: bool = False) -> subprocess.CompletedProcess:
    stub = [str(tmp / "stub.py"), str(ROOT), "1"] + (["fail"] if fail else [])
    cmd = [PY, str(ROOT / "scripts" / "knob_run.py"), *knob_args, "--", *stub] if knob_args != ("plain",) else [PY, *stub]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp)


def stub_out(p: subprocess.CompletedProcess) -> dict:
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("STUB ")]
    assert p.returncode == 0 and len(lines) == 1, (p.returncode, p.stdout, p.stderr)
    return json.loads(lines[0][5:])


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "stub.py").write_text(STUB)

    for bad, why in ((["--neg-sampling", "random", str(tmp / "stub.py")], "put -- before"), (["--neg-sampling", "random", "--"], "no script")):
        p = subprocess.run([PY, str(ROOT / "scripts" / "knob_run.py"), *bad], capture_output=True, text=True, cwd=tmp)
        assert p.returncode != 0 and why in p.stdout + p.stderr, (bad, p.stdout, p.stderr)
    for flag, vals in (("--neg-sampling", ["hard", "mix:1.5", "mix:0", "mix:x"]), ("--clause-drop", ["1.0", "-0.1", "abc"])):
        for v in vals:
            p = run(tmp, flag, v)
            assert p.returncode == 2 and "STUB" not in p.stdout, (flag, v, p.returncode, p.stdout, p.stderr)
    (tmp / "taken.npz").write_bytes(b"x")
    p = run(tmp, "--preds-out", str(tmp / "taken.npz"))
    assert p.returncode != 0 and "exists; not overwriting" in p.stdout + p.stderr and "STUB" not in p.stdout, (p.stdout, p.stderr)
    print("ok refusals: no --, no script, 4 bad --neg-sampling, 3 bad --clause-drop, an existing --preds-out (the target did not run)")

    plain = stub_out(run(tmp, "plain"))
    assert plain["cfg"]["neg_sampling"] == "hardest" and plain["cfg"]["clause_drop"] == 0.0 and len(plain["val"]) == 2
    off = stub_out(run(tmp, "--clause-drop", "0.0", "--neg-sampling", "hardest"))
    assert off == plain, (off, plain)
    p = run(tmp, "--preds-out", str(tmp / "preds_off.npz"))
    assert stub_out(p) == plain and "wrote" in p.stdout, p.stdout
    with np.load(tmp / "preds_off.npz") as z:
        preds, yv = z["preds"], z["y_val"]
        assert json.loads(str(z["knobs"])) == {} and preds.dtype == np.int16 and preds.shape == (3, 15), preds.shape
    assert preds[-1].tolist() == plain["pred"] and yv.tolist() == plain["yv"]
    for k in range(2):  # tm.fit: float((pred == yv).float().mean()) in float32
        acc = float(np.float32((preds[k] == yv).sum()) / np.float32(len(yv)))
        assert acc == plain["val"][k], (k, acc, plain["val"][k])
    explicit = stub_out(run(tmp, "--neg-sampling", "hardest"))
    bare = stub_out(run(tmp))
    assert explicit == plain and bare == plain
    print("ok knob-off identity: no knob (with and without --preds-out) and explicit default knobs print exactly the plain run; "
          "the preds file holds 2 evaluations + the final call, equal to the stub's predictions, labels and history")

    on = stub_out(run(tmp, "--neg-sampling", "random", "--clause-drop", "0.5"))
    assert on["cfg"]["neg_sampling"] == "random" and on["cfg"]["clause_drop"] == 0.5, on["cfg"]
    assert (on["ta"], on["w"]) != (plain["ta"], plain["w"]), "the knobs did not change the trained machine"
    mix = stub_out(run(tmp, "--neg-sampling", "mix:0.5"))
    assert mix["cfg"]["neg_sampling"] == "mix:0.5" and mix["cfg"]["clause_drop"] == 0.0, mix["cfg"]
    print("ok knobs on: random + clause_drop 0.5 reach the cfg and change the machine; mix:0.5 reaches the cfg")

    p = run(tmp, "--preds-out", str(tmp / "preds_fail.npz"), fail=True)
    assert p.returncode == 3 and not (tmp / "preds_fail.npz").exists(), (p.returncode, p.stdout, p.stderr)
    print("ok failing target: knob_run exits 3 and writes no preds file")
    print("PASSED")


if __name__ == "__main__":
    main()
