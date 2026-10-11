"""M5 step 1, Gate-0 diagnosis setup record: torch backend on the CPU.

Diagnosis step of 2026-10-02 21:02 UTC, after the numba Gate 0
failed: the same Gate-0 config is run again with --backend torch on the CPU. This records:
  pip_freeze.txt          `uv pip freeze` after installing torch (CPU wheel index)
  torch_env_check.txt     torch version, CUDA availability, thread counts under the run env
  train_controller.patch  full unified diff, reference snapshot -> work copy (patches 1 and 2)
  timing_probe.txt        time of the repo's torch_backend.train on a few minibatches
                          (TRAIN rows only; no val or test data), plus the extrapolated
                          per-run time for 40 epochs (limit: more than 45 min per run is unworkable)
It does NOT touch the lexical .npz (its sha256 is recorded by the m5_setup_sorted run).
Usage: .venv-cdllm/bin/python src/m5_setup_torch.py
"""
import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import Run, sha256  # noqa: E402

LL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = f"{LL}/references/code/Clause-Driven-LLM"
WORK = f"{LL}/work/Clause-Driven-LLM"
PY = f"{LL}/.venv-cdllm/bin/python"
UV = os.environ.get("UV", "uv")
NPZ = f"{LL}/work/m5/emb_sorted/clinc150__lexical.npz"
RUN_ENV = {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
N_PROBE_MB = 12

PROBE = r'''
import sys, time, numpy as np, torch
sys.path.insert(0, "%(work)s/src")
from cdllm.features.binarize import Booleanizer
from cdllm.tm import torch_backend
from cdllm.tm.coalesced import CoalescedTsetlinMachine
d = np.load("%(npz)s", allow_pickle=True)
X = Booleanizer(n_bins=2).fit_transform(d["Xtrain"]).astype(np.uint8); y = d["ytrain"].astype(np.int64)
tm = CoalescedTsetlinMachine(n_clauses=4000, T=4000, s=8.0, n_states=256, epochs=1, seed=42, batch_size=128)
rng = np.random.default_rng(42); f = X.shape[1]
ta = (256 + rng.integers(0, 2, size=(4000, 2 * f))).astype(np.int32)
w = (2 * rng.integers(0, 2, size=(4000, int(y.max()) + 1)) - 1).astype(np.int32)
nb = %(nmb)d
for k in (2, nb):  # time k minibatches (one epoch over the first k*128 TRAIN rows)
    t0 = time.time()
    torch_backend.train(ta.copy(), w.copy(), X[:128 * k], y[:128 * k], n_states=256, T=4000, s=8.0, epochs=1,
                        batch_size=128, seed=42, neg_sampling="hardest", device="cpu")
    print(f"minibatches {k}: {time.time() - t0:.2f} s", flush=True)
# cost of one validation pass (3,000 rows), timed on TRAIN rows (no val data touched)
incl_state = torch.as_tensor(ta); w_t = torch.as_tensor(w)
t0 = time.time()
torch_backend._val_acc_gpu(incl_state, w_t, X[:3000], 256, y[:3000], torch.device("cpu"))
print(f"val-pass (3000 train rows): {time.time() - t0:.2f} s", flush=True)
print("torch threads", torch.get_num_threads(), flush=True)
'''


def sh(cmd, env=None, cwd=None):
    p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def main():
    env = {**os.environ, **RUN_ENV, "PYTHONPATH": f"{WORK}/src"}
    config = {"purpose": "M5 step 1 Gate-0 diagnosis setup (torch backend on CPU)",
              "instruction": "diagnosis step of 2026-10-02 21:02 UTC: the label-order retry is stopped; torch is installed "
                             "CPU-only; Gate 0 (s 8, seed 42, sorted, locked settings) is run again with --backend torch "
                             "on CPU; pass iff test acc 0.82778 +- 0.005; a run of more than 45 min counts as unworkable; "
                             "a passing run of more than 15 min is noted with its time before the 11 grid runs; no GPU.",
              "prereg_amendment": "not written when this step first ran (2026-10-02 21:05 UTC); added to docs/prereg_M5_step1.md at 21:11 UTC",
              "torch_install": "uv pip install torch --index-url https://download.pytorch.org/whl/cpu (21:04 UTC)",
              "run_env": RUN_ENV, "nice": 19, "n_probe_minibatches": N_PROBE_MB}
    run = Run("m5_setup_torchcpu", config, data_files=[NPZ], script=__file__)
    d = run.dir
    m = {"npz_sha256": sha256(NPZ)}

    rc, out, err = sh([UV, "pip", "freeze", "--python", PY])
    open(f"{d}/pip_freeze.txt", "w").write(out + err)
    m["torch_line"] = [l for l in out.splitlines() if l.startswith("torch")]

    chk = ("import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), "
           "'cuda_build', torch.version.cuda); print('intra', torch.get_num_threads(), 'interop', "
           "torch.get_num_interop_threads()); print(torch.__config__.parallel_info())")
    rc, out, err = sh([PY, "-c", chk], env=env)
    open(f"{d}/torch_env_check.txt", "w").write(out + err)
    run.print("torch env:", " | ".join(out.strip().splitlines()[:2]))

    rc, out, _ = sh(["diff", "-u", f"{REF}/scripts/train_controller.py", f"{WORK}/scripts/train_controller.py"])
    open(f"{d}/train_controller.patch", "w").write(out)
    m["train_controller_sha256"] = sha256(f"{WORK}/scripts/train_controller.py")
    m["patch_added_lines"] = sum(1 for l in out.splitlines() if l.startswith("+") and not l.startswith("+++"))
    m["patch_removed_lines"] = sum(1 for l in out.splitlines() if l.startswith("-") and not l.startswith("---"))
    m["torch_backend_identical_to_reference"] = (sha256(f"{WORK}/src/cdllm/tm/torch_backend.py")
                                                 == sha256(f"{REF}/src/cdllm/tm/torch_backend.py"))

    rc, out, err = sh(["nice", "-n", "19", PY, "-c", PROBE % {"work": WORK, "npz": NPZ, "nmb": N_PROBE_MB}],
                      env=env, cwd=WORK)
    open(f"{d}/timing_probe.txt", "w").write(out + "\n--- stderr ---\n" + err)
    run.print("probe:", out.strip().replace("\n", " | "))
    m["probe_rc"] = rc
    try:
        t = {}
        for line in out.splitlines():
            if line.startswith("minibatches"):
                k, v = line.split(":")
                t[int(k.split()[1])] = float(v.split()[0])
            if line.startswith("val-pass"):
                t["val"] = float(line.split(":")[1].split()[0])
        per_mb = (t[N_PROBE_MB] - t[2]) / (N_PROBE_MB - 2)
        est_s = 40 * 118 * per_mb + 21 * t["val"]  # 15,000/128 -> 118 minibatches; 21 val passes
        m.update({"probe_s_per_minibatch": per_mb, "probe_val_pass_s": t["val"],
                  "estimated_run_min": est_s / 60, "over_45_min": est_s / 60 > 45, "over_15_min": est_s / 60 > 15})
        run.print(f"estimate: {per_mb:.2f} s/minibatch -> {est_s / 60:.1f} min per 40-epoch run")
    except Exception as e:  # keep the record even if parsing fails
        m["probe_parse_error"] = repr(e)
    run.finish(m)


if __name__ == "__main__":
    main()
