"""M5 step 1 setup record (docs/prereg_M5_step1.md).

Creates the first results folder for M5 step 1 and stores, as files:
  pip_freeze.txt          `uv pip freeze` of .venv-cdllm
  env_check.txt           python version, numba / BLAS thread counts under the run env
  pytest_test_data.txt    `pytest tests/test_data.py` output (new venv)
  train_controller.patch  unified diff, reference snapshot -> work copy
  work_vs_snapshot.txt    `diff -rq` reference snapshot vs work copy
  cdllm_data__init__.py   copy of the rebuilt loader
  precompute_*.txt        stdout/stderr of the repo's own lexical precompute
  vocab_cutoff.json       diagnostic: ties at the max_features=2000 cutoff
Data hashes (data_full.json and the lexical .npz) go to data_hashes.json / metrics.

Usage: .venv-cdllm/bin/python src/m5_setup.py [--label-order sorted]
"""
import argparse, json, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import Run, sha256  # noqa: E402

LL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = f"{LL}/references/code/Clause-Driven-LLM"
WORK = f"{LL}/work/Clause-Driven-LLM"
PY = f"{LL}/.venv-cdllm/bin/python"
UV = os.environ.get("UV", "uv")
DATA = f"{WORK}/data/clinc150_data_full.json"
URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
RUN_ENV = {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def sh(cmd, env=None, cwd=None):
    p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-order", default="sorted", choices=["sorted", "first_appearance"])
    a = ap.parse_args()
    emb_dir = f"{LL}/work/m5/emb_{a.label_order}"
    env = {**os.environ, **RUN_ENV, "PYTHONPATH": f"{WORK}/src", "CDLLM_LABEL_ORDER": a.label_order}

    config = {"purpose": "M5 step 1 setup record", "prereg": "docs/prereg_M5_step1.md",
              "work_copy": WORK, "reference_snapshot": REF, "venv_python": PY,
              "data_url": URL, "data_path": DATA, "data_downloaded_utc": "2026-10-02T19:49:03Z (curl -fsSL)",
              "label_order": a.label_order, "emb_dir": emb_dir, "run_env": RUN_ENV, "nice": 19}
    run = Run(f"m5_setup_{a.label_order}", config, data_files=[DATA], script=__file__)
    d = run.dir
    m = {"data_sha256": sha256(DATA), "data_bytes": os.path.getsize(DATA)}

    rc, out, err = sh([UV, "pip", "freeze", "--python", PY])
    open(f"{d}/pip_freeze.txt", "w").write(out + err)
    m["pip_freeze_rc"] = rc
    run.print("pip freeze:", out.strip().replace("\n", "; "))

    chk = ("import sys, numpy, numba, sklearn, scipy; from threadpoolctl import threadpool_info; "
           "print('python', sys.version.split()[0]); print('numpy', numpy.__version__); "
           "print('numba', numba.__version__, 'threads', numba.config.NUMBA_NUM_THREADS); "
           "print('sklearn', sklearn.__version__, 'scipy', scipy.__version__); "
           "print([(i['internal_api'], i['num_threads']) for i in threadpool_info()])")
    rc, out, err = sh([PY, "-c", chk], env=env)
    open(f"{d}/env_check.txt", "w").write(out + err)
    run.print("env check:", out.strip().replace("\n", " | "))

    rc, out, err = sh([PY, "-m", "pytest", "tests/test_data.py", "-v", "-rs"], env=env, cwd=WORK)
    open(f"{d}/pytest_test_data.txt", "w").write(out + err)
    m["pytest_test_data_rc"] = rc
    run.print("pytest tail:", out.strip().splitlines()[-1] if out.strip() else err)

    rc, out, _ = sh(["diff", "-u", f"{REF}/scripts/train_controller.py", f"{WORK}/scripts/train_controller.py"])
    open(f"{d}/train_controller.patch", "w").write(out)
    m["patch_added_lines"] = sum(1 for l in out.splitlines() if l.startswith("+") and not l.startswith("+++"))
    m["patch_removed_lines"] = sum(1 for l in out.splitlines() if l.startswith("-") and not l.startswith("---"))
    rc, out, _ = sh(["diff", "-rq", REF, WORK])
    open(f"{d}/work_vs_snapshot.txt", "w").write(out)
    shutil.copy(f"{WORK}/src/cdllm/data/__init__.py", f"{d}/cdllm_data__init__.py")
    m["train_controller_sha256"] = sha256(f"{WORK}/scripts/train_controller.py")
    m["data_module_sha256"] = sha256(f"{WORK}/src/cdllm/data/__init__.py")

    # The repo's own lexical path (run_all_seeds.sh step 2): precompute the presence features.
    cmd = ["nice", "-n", "19", PY, f"{WORK}/scripts/precompute_embeddings.py", "--dataset", "clinc150",
           "--featurizer", "lexical", "--data-dir", f"{WORK}/data", "--out-dir", emb_dir]
    rc, out, err = sh(cmd, env=env, cwd=WORK)
    open(f"{d}/precompute_stdout.txt", "w").write(" ".join(cmd) + "\n" + out)
    open(f"{d}/precompute_stderr.txt", "w").write(err)
    m["precompute_rc"] = rc
    run.print("precompute:", out.strip().replace("\n", " | "))
    if rc != 0:
        run.print("PRECOMPUTE FAILED:", err[-2000:])
        run.finish(m)
        sys.exit(1)

    import numpy as np
    npz = f"{emb_dir}/clinc150__lexical.npz"
    z = np.load(npz, allow_pickle=True)
    m["npz_path"] = npz
    m["npz_sha256"] = sha256(npz)
    m["npz_shapes"] = {k: list(z[k].shape) for k in z.files}
    m["npz_featurizer"] = str(z["featurizer"])
    intents = [str(x) for x in z["intents"]]
    m["intents_first5"] = intents[:5]
    m["intents_sorted"] = intents == sorted(intents)
    m["labels_ok"] = bool(z["ytrain"].min() == 0 and z["ytrain"].max() == 149 and len(set(z["yval"].tolist())) == 150)

    # Diagnostic only: ties at the max_features cutoff (sklearn picks the top 2000
    # by document frequency with an unstable argsort, so ties straddling the cutoff
    # can resolve differently across numpy versions / CPUs).
    sys.path.insert(0, f"{WORK}/src")
    from cdllm.data import load_clinc150
    from sklearn.feature_extraction.text import CountVectorizer
    ds = load_clinc150(DATA, label_order=a.label_order)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True).fit(ds.train_texts)
    dfreq = np.asarray(cv.transform(ds.train_texts).sum(axis=0)).ravel()
    order = np.sort(dfreq)[::-1]
    cut = int(order[1999])
    vocab_cut = {"n_terms_min_df2": int(len(dfreq)), "df_at_rank_2000": cut,
                 "n_terms_df_gt_cut": int((dfreq > cut).sum()), "n_terms_df_eq_cut": int((dfreq == cut).sum()),
                 "n_tied_terms_selected": int(2000 - (dfreq > cut).sum())}
    json.dump(vocab_cut, open(f"{d}/vocab_cutoff.json", "w"), indent=1)
    m["vocab_cutoff"] = vocab_cut
    run.print("vocab cutoff:", vocab_cut)
    run.finish(m)


if __name__ == "__main__":
    main()
