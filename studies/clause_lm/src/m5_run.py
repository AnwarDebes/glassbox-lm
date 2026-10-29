"""Run ONE M5 step 1 training (docs/prereg_M5_step1.md) inside a runlog Run.

Wraps work/Clause-Driven-LLM/scripts/train_controller.py as patched in the work copy
(the validation-accuracy lines; for --backend torch and torch_gpu also the --device
argument of the second patch) with the locked settings. The default backend is numba
on the CPU: CUDA_VISIBLE_DEVICES="", --backend numba, nice -n 19, NUMBA/OMP/MKL/OPENBLAS
threads = 2; the torch backends are described below. Records config, data hashes
(data_full.json + the lexical .npz), stdout/stderr, the run's metrics.json and
wall time in results/<UTC>_m5_.../.
(This paragraph was rewritten on 2026-10-25; the copies of this file in the run folders
of 2026-10-14 hold the earlier wording. The code is unchanged.)

No reruns: refuses if the run's out-dir already exists, unless --after-crash
is given (a rerun after a crash).

Usage: .venv-cdllm/bin/python src/m5_run.py --s 8 --seed 42 [--gate0] [--label-order sorted] [--backend numba|torch]
--backend torch (Gate-0 diagnosis, 2026-10-02 21:02 UTC) runs the repo's torch backend on the CPU
(adds --device cpu; needs train_controller.py patch 2); numba runs keep their original command and paths.
--backend torch_gpu (prereg amendment 2026-10-02 21:11 UTC) runs the repo's torch backend on the GPU, the backend
the recorded Clause-Driven-LLM results used (upstream paper/report.tex:303). It uses .venv-cdllm-gpu (torch 2.6.0+cu124, sm_70 built)
and the tm_language_model chain timeout 15m -> mem_guard 12 GiB -> gpu_run.sh exclusive -> nice 19. It shares the GPU with the
hierarchical_tm study and is started only when the queue of that study is empty.
scripts/mem_guard.py is started by its '#!/usr/bin/env python3' line, so torch_gpu needs a python3 first on PATH
(the runs of 2026-10-14 put a link there; src/m5_gpu_all.sh names the launcher). The code of this file was left as
it ran; src/launch_gpu.sh, which other GPU jobs use, names its interpreters since 2026-10-25.
"""
import argparse, json, os, platform, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import Run, sha256  # noqa: E402

LL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = f"{LL}/work/Clause-Driven-LLM"
REF = f"{LL}/references/code/Clause-Driven-LLM"
PY = f"{LL}/.venv-cdllm/bin/python"
PY_GPU = f"{LL}/.venv-cdllm-gpu/bin/python"
TMLM = os.path.join(os.path.dirname(LL), "tm_language_model")
GPU_CHAIN = ["timeout", "--kill-after=60", "15m", f"{TMLM}/scripts/mem_guard.py", "--scope", "group",
             "--limit-gib", "12", "--", f"{TMLM}/scripts/gpu_run.sh", "exclusive", "--"]
DATA = f"{WORK}/data/clinc150_data_full.json"
RUN_ENV = {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
LOCKED = {"dataset": "clinc150", "featurizer": "lexical", "n_bins": 2, "n_clauses": 4000, "T": 4000,
          "n_states": 256, "epochs": 40, "batch_size": 128, "neg_sampling": "hardest", "backend": "numba"}
GATE0_TARGET = 0.8277777777777777  # results/tables/headline.json of the upstream Clause-Driven-LLM repository, clinc150__lexical
GATE0_TOL = 0.005


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label-order", default="sorted", choices=["sorted", "first_appearance"])
    ap.add_argument("--gate0", action="store_true", help="evaluate the Gate-0 rule on this run")
    ap.add_argument("--after-crash", action="store_true", help="rerun only after a crash (report it)")
    ap.add_argument("--backend", default="numba", choices=["numba", "torch", "torch_gpu"])
    a = ap.parse_args()

    s_tag = f"{a.s:g}"
    emb_dir = f"{LL}/work/m5/emb_{a.label_order}"
    npz = f"{emb_dir}/clinc150__lexical.npz"
    bsuf = {"numba": "", "torch": "_torchcpu", "torch_gpu": "_torchgpu"}[a.backend]
    out_dir = f"{LL}/work/m5/runs/{a.label_order}{bsuf}/s{s_tag}_seed{a.seed}"
    run_sub = f"{out_dir}/{LOCKED['dataset']}__{LOCKED['featurizer']}__seed{a.seed}"
    if os.path.exists(f"{run_sub}/metrics.json"):
        sys.exit(f"REFUSED: {run_sub}/metrics.json exists; each run is scored on test once.")
    if os.path.exists(out_dir) and not a.after_crash:
        sys.exit(f"REFUSED: {out_dir} exists (partial run?). Use --after-crash only after a crash.")
    if not os.path.exists(npz):
        sys.exit(f"missing {npz}: run src/m5_setup.py --label-order {a.label_order} first")

    gpu = a.backend == "torch_gpu"
    L = {**LOCKED, "backend": a.backend}
    cmd = (GPU_CHAIN if gpu else []) + [
           "nice", "-n", "19", PY_GPU if gpu else PY, f"{WORK}/scripts/train_controller.py",
           "--dataset", L["dataset"], "--featurizer", L["featurizer"], "--seed", str(a.seed),
           "--n-bins", str(L["n_bins"]), "--n-clauses", str(L["n_clauses"]), "--T", str(L["T"]),
           "--s", str(a.s), "--n-states", str(L["n_states"]), "--epochs", str(L["epochs"]),
           "--batch-size", str(L["batch_size"]), "--neg-sampling", L["neg_sampling"],
           "--backend", "torch" if gpu else L["backend"], "--emb-dir", emb_dir, "--out-dir", out_dir]
    if a.backend == "torch":
        cmd += ["--device", "cpu"]
    if gpu:
        cmd += ["--device", "cuda"]
    run_env = {**RUN_ENV, "CUDA_VISIBLE_DEVICES": "0"} if gpu else RUN_ENV
    env = {**os.environ, **run_env, "PYTHONPATH": f"{WORK}/src"}

    tm_src = f"{WORK}/src/cdllm/tm/coalesced.py"
    config = {**L, "s": a.s, "seed": a.seed, "label_order": a.label_order, "gate0": a.gate0,
              "after_crash": a.after_crash, "cmd": cmd, "run_env": run_env, "nice": 19,
              "emb_dir": emb_dir, "out_dir": out_dir, "venv_python": PY_GPU if gpu else PY,
              "train_controller_sha256": sha256(f"{WORK}/scripts/train_controller.py"),
              "data_module_sha256": sha256(f"{WORK}/src/cdllm/data/__init__.py"),
              "coalesced_sha256": sha256(tm_src),
              "coalesced_identical_to_reference": sha256(tm_src) == sha256(f"{REF}/src/cdllm/tm/coalesced.py"),
              "host": platform.node(), "nproc": os.cpu_count(), "loadavg_at_start": os.getloadavg(),
              "prereg": "docs/prereg_M5_step1.md"}
    if a.backend == "torch":
        import torch
        config.update({"torch_version": torch.__version__, "device": "cpu",
                       "torch_cuda_available_in_driver": torch.cuda.is_available()})
    if gpu:  # versions read in a child process of the GPU venv; this wrapper never touches CUDA
        probe = subprocess.run([PY_GPU, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                               capture_output=True, text=True).stdout.split()
        config.update({"torch_version": probe[0] if probe else None, "torch_cuda_build": probe[1] if len(probe) > 1 else None,
                       "device": "cuda", "gpu_chain": GPU_CHAIN})
    name = f"m5_{'gate0_' if a.gate0 else ''}s{s_tag}_seed{a.seed}_{a.label_order}{bsuf}"
    run = Run(name, config, data_files=[DATA, npz], script=__file__)
    shutil.copy(f"{WORK}/scripts/train_controller.py", f"{run.dir}/train_controller.py")
    shutil.copy(f"{WORK}/src/cdllm/data/__init__.py", f"{run.dir}/cdllm_data__init__.py")
    run.print("cmd:", " ".join(cmd))

    start = utc()
    t0 = time.time()
    with open(f"{run.dir}/stdout.txt", "w") as fo, open(f"{run.dir}/stderr.txt", "w") as fe:
        rc = subprocess.run(cmd, env=env, cwd=WORK, stdout=fo, stderr=fe).returncode
    proc_wall = time.time() - t0
    end = utc()
    run.print(f"returncode {rc}; start {start}; end {end}; process wall {proc_wall:.1f}s")
    err = open(f"{run.dir}/stderr.txt").read()
    if err.strip():
        run.print("stderr tail:", err[-1500:])

    summary = {"returncode": rc, "start_utc": start, "end_utc": end, "process_wall_s": round(proc_wall, 1),
               "s": a.s, "seed": a.seed, "label_order": a.label_order}
    if rc != 0 or not os.path.exists(f"{run_sub}/metrics.json"):
        summary["CRASH"] = True
        run.finish(summary)
        sys.exit(1)

    tm = json.load(open(f"{run_sub}/metrics.json"))
    shutil.copy(f"{run_sub}/metrics.json", f"{run.dir}/train_metrics.json")
    keys = ["test_accuracy", "val_accuracy", "best_val_acc", "test_macro_f1", "oos_auroc",
            "oos_recall_at_95prec", "selective_acc_at_80cov", "aurc", "train_seconds",
            "n_literals", "model_version_id", "clause_stats", "config"]
    summary.update({k: tm.get(k) for k in keys})
    hist = tm.get("history", [])
    if hist:
        best = max(hist, key=lambda h: h["val_acc"])  # first maximum = the restored snapshot
        summary["best_epoch"] = best["epoch"]
        summary["val_history"] = [round(h["val_acc"], 4) for h in hist]
    summary["val_equals_best_val_acc"] = tm.get("val_accuracy") == tm.get("best_val_acc")
    if a.gate0:
        diff = abs(tm["test_accuracy"] - GATE0_TARGET)
        summary.update({"gate0_target": GATE0_TARGET, "gate0_tol": GATE0_TOL, "gate0_abs_diff": diff,
                        "gate0_pass": diff <= GATE0_TOL, "gate0_exact_match": tm["test_accuracy"] == GATE0_TARGET})
    run.finish(summary)


if __name__ == "__main__":
    main()
