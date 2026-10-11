"""Logistic-regression baseline for E5a (design decision 'news topics'; context for gates.md Check 2c).
The same 1,024 word-presence features as the TM, trained on train (60,000), scored on dev (10,000) with the TM's
metrics (text_driver.metrics). C is chosen on dev from a small grid; every value is reported. CPU only (.venv-tm).
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_driver import DATA, load_split, metrics  # noqa: E402


def main():
    Xtr, ytr, _ = load_split(DATA, "train")
    Xdv, ydv, _ = load_split(DATA, "dev")
    Xtr, Xdv, res = Xtr.astype(np.float64), Xdv.astype(np.float64), {}
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        res[str(C)] = {"train_acc": float(m.score(Xtr, ytr)), "n_iter": int(m.n_iter_.max()),
                       **{f"dev_{k}": v for k, v in metrics(m.decision_function(Xdv).T, ydv).items()}}
        print("LR", json.dumps({"C": C, **res[str(C)]}), flush=True)
    best = max(res, key=lambda c: (res[c]["dev_acc"], -float(c)))  # ties: the smaller C (stronger regularisation)
    out = {"by_C": res, "best_C_by_dev_acc": float(best), "best": res[best]}
    print("FINAL", json.dumps(out["best"] | {"C": float(best)}), flush=True)
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        json.dump({k: {"value": v, "label": "measured"} for k, v in out.items()}, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
