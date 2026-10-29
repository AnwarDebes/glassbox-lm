#!/usr/bin/env python
"""WebGuard veto test, the scoring (webguard/PLAN_wg_veto.md "Bar and metrics"; reads the TEST part once).

    python -I wg_eval.py

From results/wg_preds.npz (saved by wg_run.py) and feats/wg_bits.npz:
- test accuracy and HIGH recall: the TM (each of seeds 1-3 and their mean) and LR; "false alarm" = a SAFE row
  predicted HIGH. LR's HIGH recall also at the TM's own false-alarm rate: the threshold on P(HIGH) is set on DEV so
  that LR's dev rate equals the TM's mean dev rate (above the threshold the row is HIGH, else the argmax of SAFE/LOW).
- intervals: 10,000 bootstrap resamples (numpy default_rng(20261022)) over the 18 test SITES, 95% percentiles.
- BAR: non-inferior when the lower end of the interval of (TM mean accuracy - LR accuracy) is above -5 points.
- the guard (e70's form): a veto clause on the veto bit with weight -M on SAFE, M = SAFE's positive weights + the
  absolute weights of HIGH + 1. With it, SAFE cannot win when the bit is on. LR: the same rule (argmax of LOW/HIGH).
  Accuracy and HIGH recall with and without it, and the number of test rows it changes.
- the solver check (scripts/verify_policy.py's build_and_solve, imported unedited; HiGHS): can SAFE win with the veto
  bit on, for the TM alone and with the guard? Domain rows: thermometer order within each concept, exactly one role
  and one tag, count bits in order.
Writes results/wg_eval.json and wg_eval.txt; refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DL / "src"))
sys.path.insert(1, str(DL / "scripts"))
import verify_policy as vp  # noqa: E402
from dl import cpu_tm_fast  # noqa: E402

SAFE, LOW, HIGH = 0, 1, 2
N_BOOT, MARGIN = 10_000, 5.0


def acc(p, y):
    return float((p == y).mean() * 100)


def high_recall(p, y):
    return float((p[y == HIGH] == HIGH).mean() * 100) if (y == HIGH).any() else float("nan")


def false_alarm(p, y):
    return float((p[y == SAFE] == HIGH).mean() * 100) if (y == SAFE).any() else float("nan")


def tm_dict(seed: int, guard: bool):
    z = np.load(R / "models" / f"tm_seed{seed}.npz")
    m = cpu_tm_fast.pack(z["ta"], z["w"])
    clauses = [m["lits"][s:e].tolist() for s, e in zip(m["starts"], list(m["starts"][1:]) + [len(m["lits"])])]
    W = m["w"].copy()
    tm = {"clauses": clauses, "W": W, "F": m["n_features"], "pack": m}
    if guard:
        M = int(np.clip(W[:, SAFE], 0, None).sum() + np.abs(W[:, HIGH]).sum() + 1)
        tm["clauses"] = clauses + [[META["veto_bit"]]]
        row = np.zeros((1, W.shape[1]), np.int64)
        row[0, SAFE] = -M
        tm["W"] = np.concatenate([W, row], 0)
        tm["M"] = M
    return tm


def domain_rows():
    rows = []
    for g in META["groups"]["thermo"] + META["groups"]["counts"]:
        for a, b in zip(g, g[1:]):  # a higher level implies the lower one: x_b - x_a <= 0
            rows.append(({b: 1.0, a: -1.0}, -np.inf, 0.0))
    for g in META["groups"]["onehot"]:
        rows.append(({i: 1.0 for i in g}, 1.0, 1.0))
    return rows


def main() -> None:
    global META
    oj, ot = R / "wg_eval.json", R / "wg_eval.txt"
    assert not oj.exists() and not ot.exists()
    META = json.loads((HERE / "feats" / "wg_bits.json").read_text())
    z = np.load(HERE / "feats" / "wg_bits.npz")
    B, y, site = z["B"], z["y"], z["site"]
    P = np.load(R / "wg_preds.npz")
    dv, te = P["dev_idx"], P["test_idx"]
    vb = META["veto_bit"]
    tm_dev = [P[f"tm{k}_dev_sums"].argmax(1) for k in (1, 2, 3)]
    tm_te = [P[f"tm{k}_test_sums"].argmax(1) for k in (1, 2, 3)]
    lr_dev_p, lr_te_p = P["lr_dev_prob"], P["lr_test_prob"]
    lr_te = lr_te_p.argmax(1)
    # LR at the TM's dev false-alarm rate (threshold on P(HIGH) chosen on dev)
    fa_tm_dev = float(np.mean([false_alarm(p, y[dv]) for p in tm_dev]))
    safe_dev = lr_dev_p[y[dv] == SAFE, HIGH]
    thr = float(np.quantile(safe_dev, 1 - fa_tm_dev / 100)) if len(safe_dev) else 1.0

    def lr_at(prob):
        p = np.where(prob[:, HIGH] >= thr, HIGH, prob[:, :HIGH].argmax(1))
        return p

    lr_te_fa = lr_at(lr_te_p)
    # the guard on the TM: -M on SAFE when the veto bit is on (M per seed), so SAFE cannot win there
    tm_te_g = []
    Ms = []
    for k in (1, 2, 3):
        s = P[f"tm{k}_test_sums"].astype(np.int64).copy()
        M = tm_dict(k, True)["M"]
        Ms.append(M)
        s[B[te, vb] == 1, SAFE] -= M
        tm_te_g.append(s.argmax(1))
    lr_te_g = np.where(B[te, vb] == 1, lr_te_p[:, LOW:].argmax(1) + LOW, lr_te)
    yt = y[te]
    res = {"test_rows": int(len(te)), "test_sites": int(len(set(site[te]))), "labels": {n: int((yt == i).sum()) for i, n in enumerate(("SAFE", "LOW", "HIGH"))},
           "tm_s": float(P["tm_s"]), "lr_C": float(P["lr_C"]), "veto_on_test": int(B[te, vb].sum())}
    res["tm"] = {"acc_per_seed": [round(acc(p, yt), 2) for p in tm_te], "acc": round(float(np.mean([acc(p, yt) for p in tm_te])), 2),
                 "high_recall": round(float(np.mean([high_recall(p, yt) for p in tm_te])), 2),
                 "false_alarm": round(float(np.mean([false_alarm(p, yt) for p in tm_te])), 2), "dev_false_alarm": round(fa_tm_dev, 2)}
    res["lr"] = {"acc": round(acc(lr_te, yt), 2), "high_recall": round(high_recall(lr_te, yt), 2), "false_alarm": round(false_alarm(lr_te, yt), 2),
                 "at_tm_dev_false_alarm": {"threshold": round(thr, 4), "acc": round(acc(lr_te_fa, yt), 2), "high_recall": round(high_recall(lr_te_fa, yt), 2),
                                           "false_alarm": round(false_alarm(lr_te_fa, yt), 2)}}
    res["guard"] = {"M_per_seed": Ms,
                    "tm": {"acc": round(float(np.mean([acc(p, yt) for p in tm_te_g])), 2), "high_recall": round(float(np.mean([high_recall(p, yt) for p in tm_te_g])), 2),
                           "rows_changed_per_seed": [int((a != b).sum()) for a, b in zip(tm_te, tm_te_g)]},
                    "lr": {"acc": round(acc(lr_te_g, yt), 2), "high_recall": round(high_recall(lr_te_g, yt), 2), "rows_changed": int((lr_te != lr_te_g).sum())}}
    # site bootstrap
    sites = np.array(sorted(set(site[te])))
    by = {s: np.flatnonzero(site[te] == s) for s in sites}
    rng = np.random.default_rng(20261022)
    diffs, tms, lrs, hrs = [], [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[s] for s in rng.choice(sites, len(sites), replace=True)])
        a_tm = np.mean([acc(p[idx], yt[idx]) for p in tm_te])
        a_lr = acc(lr_te[idx], yt[idx])
        diffs.append(a_tm - a_lr)
        tms.append(a_tm)
        lrs.append(a_lr)
        hrs.append(np.mean([high_recall(p[idx], yt[idx]) for p in tm_te]) - high_recall(lr_te[idx], yt[idx]))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 2), round(float(np.nanpercentile(v, 97.5)), 2)]  # noqa: E731
    res["bootstrap"] = {"resamples": N_BOOT, "tm_acc_ci": ci(tms), "lr_acc_ci": ci(lrs), "diff_acc": round(res["tm"]["acc"] - res["lr"]["acc"], 2),
                        "diff_acc_ci": ci(diffs), "diff_high_recall_ci": ci(hrs)}
    res["bar"] = {"rule": f"lower end of the site-bootstrap 95% interval of (TM - LR) accuracy > -{MARGIN}",
                  "lower": res["bootstrap"]["diff_acc_ci"][0], "non_inferior": bool(res["bootstrap"]["diff_acc_ci"][0] > -MARGIN)}
    # solver check: can SAFE win with the veto bit on?
    rows_dom = domain_rows()
    res["solver"] = []
    for k in (1, 2, 3):
        for guard in (False, True):
            tm = tm_dict(k, guard)
            out = vp.build_and_solve(tm, rows_dom, {vb: 1}, SAFE, [LOW, HIGH], time_limit=600.0, threads=2)
            rec = {"seed": k, "guard": guard, "status": out["status"], "seconds": out.get("seconds"), "n_active_clauses": out.get("n_active_clauses")}
            if out.get("status") == "feasible":
                x = out["x"]
                sc = vp.tm_scores_fixed(tm, x)
                rec["counterexample_scores"] = [int(v) for v in sc]
                rec["counterexample_on_bits"] = [META["names"][i] for i in np.flatnonzero(x)][:40]
            res["solver"].append(rec)
            print(json.dumps(rec), flush=True)
    oj.write_text(json.dumps(res, indent=1))
    t, lr, g, bs = res["tm"], res["lr"], res["guard"], res["bootstrap"]
    lines = [f"WebGuard veto test (our site split; {res['test_rows']} test rows from {res['test_sites']} sites; a first guard check, not a benchmark)",
             f"TM (s {res['tm_s']:g}, seeds 1-3): accuracy {t['acc']:.2f} {t['acc_per_seed']}, HIGH recall {t['high_recall']:.2f}, false alarm {t['false_alarm']:.2f}",
             f"LR (C {res['lr_C']:g}): accuracy {lr['acc']:.2f}, HIGH recall {lr['high_recall']:.2f}, false alarm {lr['false_alarm']:.2f}; at the TM's dev false-alarm rate: "
             f"HIGH recall {lr['at_tm_dev_false_alarm']['high_recall']:.2f}, accuracy {lr['at_tm_dev_false_alarm']['acc']:.2f}",
             f"TM - LR accuracy {bs['diff_acc']:+.2f}, site-bootstrap 95% interval {bs['diff_acc_ci']}; BAR (lower end > -{MARGIN:g}): "
             f"{'NON-INFERIOR' if res['bar']['non_inferior'] else 'NOT SHOWN'}",
             f"guard (veto bit on {res['veto_on_test']} test rows): TM accuracy {g['tm']['acc']:.2f}, HIGH recall {g['tm']['high_recall']:.2f}, rows changed {g['tm']['rows_changed_per_seed']}; "
             f"LR accuracy {g['lr']['acc']:.2f}, HIGH recall {g['lr']['high_recall']:.2f}, rows changed {g['lr']['rows_changed']}",
             "solver (can SAFE win with the veto bit on?): " + "; ".join(f"seed {r['seed']} {'guarded' if r['guard'] else 'alone'}: {r['status']}" for r in res["solver"])]
    ot.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
