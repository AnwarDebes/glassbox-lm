"""T0 arm 4: multinomial logistic regression on the clause outputs of the trained bank.

    python -m tally.t0.logreg <final run folder>   (torch env, GPU, under gpu_run.sh exclusive)

Inputs: fired_<split>.npz of a final run (packed clause outputs, class labels, words).
Weights with bias, full-batch L-BFGS on train; L2 strength chosen on calib by log-loss from
L2_GRID; one temperature is not needed (the model is fitted by log-loss). Writes its own run
folder with per-position bits for calib, valid and devtest (full distribution over V, same
"other" spread as every arm) and checks sum-to-one on 1,000 positions per split.
"""
import json
import os
import sys

import numpy as np
import torch

from tally import runlog
from tally.t0 import data

L2_GRID = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
MAX_ITER = 300
SPLITS = ("calib", "valid", "devtest")


def load(src, sp, dev):
    z = np.load(os.path.join(src, f"fired_{sp}.npz"))
    C = int(z["C"])
    X = torch.from_numpy(np.unpackbits(z["bits"], axis=1)[:, :C]).to(dev)  # uint8 on the GPU
    return X, torch.from_numpy(z["y"].astype(np.int64)).to(dev), z["words"]


def nll(model, X, y, chunk=100000):
    tot = 0.0
    for lo in range(0, len(y), chunk):
        z = model(X[lo:lo + chunk].float())
        tot = tot + torch.nn.functional.cross_entropy(z, y[lo:lo + chunk], reduction="sum")
    return tot / len(y)


def fit(X, y, K, l2, run):
    torch.manual_seed(0)
    model = torch.nn.Linear(X.shape[1], K).to(X.device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=MAX_ITER, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
    state = {}

    def closure():
        opt.zero_grad()
        loss = nll(model, X, y) + l2 * (model.weight ** 2).sum()
        loss.backward()
        state["loss"] = float(loss)
        return loss

    opt.step(closure)
    run.print(f"l2={l2} train_objective={state['loss']:.5f} iters={opt.state[opt._params[0]]['n_iter']}")
    return model


def word_bits(model, X, words, cls, u):
    with torch.no_grad():
        z = np.concatenate([model(X[lo:lo + 100000].float()).cpu().numpy()
                            for lo in range(0, X.shape[0], 100000)]).astype(np.float64)
    m = z.max(1, keepdims=True)
    lp = z - m - np.log(np.exp(z - m).sum(1, keepdims=True))
    c = cls[words]
    out = -lp[np.arange(len(words)), c] / np.log(2)
    oth = c == data.K_TOP
    out[oth] += -np.log2(u[words[oth]])
    rows = np.random.default_rng(0).choice(len(words), 1000, replace=False)
    P = np.exp(lp[rows])
    full = np.where((cls < data.K_TOP)[None, :], P[:, np.minimum(cls, data.K_TOP)], P[:, [data.K_TOP]] * u[None, :])
    ok = bool(np.all(np.abs(full.sum(1) - 1) < 1e-6) and full.min() > 0)
    return out, ok


def main():
    src = os.path.abspath(sys.argv[1])
    dev = torch.device("cuda")
    cfg = dict(source_run=os.path.basename(src), l2_grid=L2_GRID, max_iter=MAX_ITER, optimizer="LBFGS strong_wolfe",
               torch=torch.__version__, device=torch.cuda.get_device_name(0))
    run = runlog.Run("t0_logreg_of_" + os.path.basename(src).split("_", 1)[1].replace("t0_", "", 1), cfg)
    d = data.build()
    K = data.K_TOP + 1
    Xtr, ytr, _ = load(src, "train", dev)
    Xc, yc, wc = load(src, "calib", dev)
    best = None
    for l2 in L2_GRID:
        m = fit(Xtr, ytr, K, l2, run)
        with torch.no_grad():
            v = float(nll(m, Xc, yc))
        run.print(f"l2={l2} calib_nll={v:.5f}")
        if best is None or v < best[1]:
            best = (l2, v, m)
    l2, v, model = best
    del Xtr
    out, metrics = {}, dict(l2=l2, calib_nll=v)
    for sp in [s for s in SPLITS if os.path.exists(os.path.join(src, f"fired_{s}.npz"))]:
        X, _, w = load(src, sp, dev)
        bits, ok = word_bits(model, X, w, d["cls"], d["u"])
        assert ok, f"distribution check failed on {sp}"
        out[f"logreg__{sp}"] = bits
        metrics[f"logreg_{sp}_bpw"] = float(bits.mean())
    np.savez_compressed(run.path("bits.npz"), **out)
    run.finish(metrics)


if __name__ == "__main__":
    main()
