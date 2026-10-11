"""Equivalence of the GPU LR backend on the logged R1 cells (docs/13 amendment A8; exploratory).

Recomputes every LR cell of a finished R1 probe run with `backend="gpu"` (tmlm.ssl.lr_gpu: sklearn's
objective and scipy L-BFGS-B call, loss and gradient on the GPU) from the same features, label draws and
folds, and compares each cell with the logged CPU cell: accuracy, chosen C and the CV accuracy of every C.
Also records the wall time per cell against the logged CPU wall time (shared GPU and a contended CPU, so
the timing is indicative).

    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/run_logged.py \
        --name ssl_r1_lr_equivalence_A3 --config experiments/cvtm/configs/ssl_r1_lr_equivalence.json -- \
        .venv-tm/bin/python -u experiments/ssl_r1_lr_equivalence.py --probe-run results/<A3 probe> \
        --features-run results/<A3 features>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssl_r1_pretext as R  # noqa: E402

from tmlm.ssl import downstream as D  # noqa: E402
from tmlm.ssl import probes as PR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(R.ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_pretext.json"))
    ap.add_argument("--probe-run", required=True)
    ap.add_argument("--features-run", required=True)
    ap.add_argument("--baselines-run", default=None)
    args = ap.parse_args()
    p = R.load_params(args.config)
    rec = R.Rec(False)
    probe_run, features_run = R.rp(args.probe_run), R.rp(args.features_run)
    store = R.Store(features_run, R.rp(args.baselines_run), p)
    rows = [json.loads(line) for line in open(probe_run / "cells.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["clf"] == "lr" and not r.get("smoke")]
    rec.m("equiv.probe_run", str(probe_run))
    rec.m("equiv.features_run", str(features_run))
    rec.m("equiv.n_cells", len(rows))
    out = R.out_dir("equiv") / "cells_gpu.jsonl"
    n_acc = n_choice = n_cv = 0
    max_dacc = 0.0
    cpu_s = gpu_s = 0.0
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            task, budget, draw, fs = r["task"], r["budget"], r["draw"], r["set"]
            code = D.TASKS[task]["code"]
            pool_y = np.asarray(store.labels[task]["pool_y"], np.int64)
            dev_y = np.asarray(store.labels[task]["dev_y"], np.int64)
            pr = PR.label_sample(pool_y, budget, code, draw)
            yl = pool_y[pr]
            t0 = time.time()
            Xl, Xd, dense = store.lr_matrices(fs, store.rows(task, pr), store.dev_rows(task))
            c = PR.lr_probe(Xl, yl, Xd, dev_y, PR.fold_seed(code, budget, draw), dense=dense, c_grid=p["c_grid"],
                            folds=int(p["folds"]), max_iter=int(p["lr_max_iter"]), backend="gpu")
            dt = time.time() - t0
            cv_logged = {float(k): v for k, v in r["cv"]["acc"].items()}
            same_acc = c.accuracy == r["acc"]
            same_choice = float(c.choice) == float(r["choice"])
            same_cv = c.cv["acc"] == cv_logged
            n_acc += same_acc
            n_choice += same_choice
            n_cv += same_cv
            max_dacc = max(max_dacc, abs(c.accuracy - r["acc"]))
            cpu_s += float(r["wall_s"])
            gpu_s += dt
            fh.write(json.dumps({"task": task, "budget": budget, "draw": draw, "set": fs, "acc_cpu": r["acc"],
                                 "acc_gpu": c.accuracy, "choice_cpu": r["choice"], "choice_gpu": c.choice,
                                 "cv_cpu": cv_logged, "cv_gpu": c.cv["acc"], "wall_cpu_s": r["wall_s"],
                                 "wall_gpu_s": dt}, default=str) + "\n")
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                rec.log(f"{i + 1}/{len(rows)} cells: identical acc {n_acc}, choice {n_choice}, cv {n_cv}; "
                        f"max |d acc| {max_dacc:.2e}; cpu {cpu_s:.0f} s vs gpu {gpu_s:.0f} s")
                rec.flush()
    rec.m("equiv.identical_accuracy", n_acc)
    rec.m("equiv.identical_choice", n_choice)
    rec.m("equiv.identical_cv", n_cv)
    rec.m("equiv.max_abs_accuracy_difference", max_dacc)
    rec.m("equiv.logged_cpu_wall_s", cpu_s, label="measured", note="from the logged cells (contended CPU)")
    rec.m("equiv.gpu_wall_s", gpu_s, label="measured", note="shared GPU lock, one CPU slot; indicative")
    rec.check("gpu_cells_identical", n_acc == len(rows) and n_choice == len(rows) and n_cv == len(rows),
              identical_accuracy=n_acc, identical_choice=n_choice, identical_cv=n_cv, n=len(rows))
    rec.m("failed_checks", rec.failed)
    rec.flush()
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
