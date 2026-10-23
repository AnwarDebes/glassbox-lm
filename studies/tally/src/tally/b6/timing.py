"""Timing run: one epoch of one member on the real train split, plus scoring cost.

    PYTHONPATH=src python -m tally.b6.timing --dataset fmnist --member thresh10 [--clauses 100]

Measures encoding, one full training pass over the train split, class sums for a sample of
rows (extrapolated to the split) and clause outputs for a sample of rows. Labelled "timing";
nothing here is a result.
"""
import argparse
import time

import numpy as np

from tally.b6 import config, data, engine
from tally.runlog import Run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(config.MEMBERS))
    ap.add_argument("--member", required=True)
    ap.add_argument("--clauses", type=int, default=config.CLAUSES)
    ap.add_argument("--rows", type=int, default=None, help="train rows (default: all)")
    ap.add_argument("--score-rows", type=int, default=2000)
    a = ap.parse_args()
    ds = a.dataset
    if a.member == "concat":
        spec = dict(config.MEMBERS[ds][0], name="concat", view="concat")
    else:
        spec = next(m for m in config.MEMBERS[ds] if m["name"] == a.member)
    cfg = dict(dataset=ds, member=spec, clauses=a.clauses, rows=a.rows, score_rows=a.score_rows)
    run = Run(f"b6timing_{ds}_{a.member}_c{a.clauses}", cfg, data.files(ds, []))
    X = data.view(ds, spec["view"], "train")
    y = data.labels(ds, "train")
    rows = np.arange(len(y)) if a.rows is None else data.subset_rows(y, a.rows, 0)
    X, y = X[rows], y[rows]
    T = config.T_of(spec, a.clauses)
    m = engine.Member(spec, config.CLASSES[ds], X, seed=1, clauses=a.clauses, T=T)
    t = time.time()
    enc = m.encode(X)
    t_enc = time.time() - t
    run.print(f"encoded {X.shape} -> {enc.shape} in {t_enc:.1f}s ({enc.nbytes / 2**20:.0f} MiB)")
    t = time.time()
    m.fit_indices(enc, y, np.arange(len(y)))
    t_fit = time.time() - t
    run.print(f"epoch 1 fit {len(y)} rows in {t_fit:.1f}s")
    t = time.time()
    m.fit_indices(enc, y, np.arange(len(y)))
    t_fit2 = time.time() - t
    run.print(f"epoch 2 fit {len(y)} rows in {t_fit2:.1f}s")
    sr = np.arange(min(a.score_rows, len(y)))
    t = time.time()
    cs = m.class_sums(enc, sr)
    t_pred = (time.time() - t) / len(sr)
    acc = float((cs.argmax(1) == y[sr]).mean())
    t = time.time()
    m.clause_outputs(enc[: min(500, len(y))])
    t_co = (time.time() - t) / min(500, len(y))
    run.finish(dict(encode_s=round(t_enc, 2), fit_epoch1_s=round(t_fit, 2), fit_epoch2_s=round(t_fit2, 2),
                    fit_ms_per_example=round(1000 * t_fit2 / len(y), 4),
                    score_ms_per_example=round(1000 * t_pred, 4),
                    clause_outputs_ms_per_example=round(1000 * t_co, 4),
                    train_rows=int(len(y)), train_acc_after_2_epochs_on_score_rows=round(acc, 4),
                    encoded_mib=round(enc.nbytes / 2**20, 1), T=T))


if __name__ == "__main__":
    main()
