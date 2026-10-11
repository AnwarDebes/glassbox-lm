"""Analyse gpu_share_test.sh.

Speed: steady-state seconds per pass = fit_s + eval_s over passes 2-4 (pass 1 includes compiling).
Throughput gain with n jobs at once = (n / mean seconds per pass) / (1 / single-job seconds per pass).
Identity: seed 1 alone vs seed 1 in each batch: per-pass dev_acc, dev_auc, pred_pos and the printed rules.
If seed 1 alone twice already differs (float32 atomicAdd with clause values up to 2^24 at A 64, K 2), the 2,000-clause
setting is the strict identity judge, and the other setting is judged against the alone-vs-alone spread.
Decision rule: switch only if gain >= 1.3 AND identical results. Conservative reading: baseline = the faster of
the two alone runs, and a gain within about 5% of 1.3x (1.235-1.365) is BORDERLINE, so one job at a time is kept.
Eval schedule: --eval-every 5 vs every pass, 2,000-clause setting, 10 passes: final metrics, rules and include bits.
"""
import glob
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def rd(name):
    d = sorted(glob.glob(os.path.join(R, f"*_{name}")))
    return d[-1] if d else None


def ep(d):
    return [json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(d, "log.txt")) if l.startswith("EPOCH ")]


def rules(d):
    p = os.path.join(d, "rules.txt")
    return open(p).read() if os.path.exists(p) else None


def spread(a, b):
    if len(a) != len(b):
        return float("inf")
    return max(max(abs(x[k] - y[k]) for k in ("dev_acc", "dev_auc", "pred_pos")) for x, y in zip(a, b))


def sec_per_pass(rows):
    rows = rows[1:] if len(rows) > 1 else rows
    return sum(r["fit_s"] + r["eval_s"] for r in rows) / len(rows)


def main():
    out = ["# GPU-sharing test result", ""]
    verdict = {}
    for setting in ("best", "granmo"):
        base, rep = rd(f"share_{setting}_n1_s1_exploratory"), rd(f"share_{setting}_n1b_s1_exploratory")
        out.append(f"## {setting}")
        if not base:
            out.append("- single-job run missing")
            continue
        b = ep(base)
        t_a = sec_per_pass(b)
        t_b = sec_per_pass(ep(rep)) if rep else t_a
        t1 = min(t_a, t_b)  # conservative baseline: the FASTER alone run
        out.append(f"- alone: {t_a:.2f} s per pass; alone again: {t_b:.2f} s per pass; baseline = faster = {t1:.2f}")
        tol = 0.0
        if rep:
            sp = spread(b, ep(rep))
            same_rules = rules(rep) == rules(base)
            out.append(f"- alone twice: max per-pass difference {sp:.6f}; rules identical: {same_rules}")
            if sp > 0 or not same_rules:
                tol = sp
                out.append("  -> the package is NOT exactly repeatable for this setting; tolerance = this spread")
        for tag, n in (("n2", 2), ("n3", 3), ("mps2", 2)):
            ds = [rd(f"share_{setting}_{tag}_s{s}_exploratory") for s in range(1, n + 1)]
            if any(d is None for d in ds):
                out.append(f"- {tag}: not run or missing")
                continue
            per = [sec_per_pass(ep(d)) for d in ds]
            gain = (n / (sum(per) / n)) / (1 / t1)
            sp = spread(b, ep(ds[0]))
            ident = sp <= tol and (tol > 0 or rules(ds[0]) == rules(base))
            # gain within about 5% of 1.3x counts as BORDERLINE and keeps one job at a time
            label = "PASS" if (gain >= 1.3 * 1.05 and ident) else ("BORDERLINE" if (ident and gain >= 1.3 * 0.95) else "FAIL")
            ok = label == "PASS"
            verdict[(setting, tag)] = label
            out.append(f"- {tag}: {', '.join(f'{p:.2f}' for p in per)} s/pass each; gain {gain:.2f}x vs the faster alone run; "
                       f"seed-1 max diff {sp:.6f} (tolerance {tol:.6f}); identical: {ident} -> {label}")
    e1, e5 = rd("evalsched_best_every1_s1_exploratory"), rd("evalsched_best_every5_s1_exploratory")
    out.append("")
    out.append("## evaluation schedule (--eval-every 5 vs every pass, 2,000 clauses, 10 passes)")
    if e1 and e5:
        f1 = next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(e1, "log.txt")) if l.startswith("FINAL "))
        f5 = next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(e5, "log.txt")) if l.startswith("FINAL "))
        same_final = all(f1[k] == f5[k] for k in ("dev_acc", "dev_auc", "pred_pos", "partner_shuffle_auc"))
        b1, b5 = np.load(os.path.join(e1, "include_bits.npz")), np.load(os.path.join(e5, "include_bits.npz"))
        same_bits = all(np.array_equal(b1[k], b5[k]) for k in ("class1", "class0"))
        w1, w5 = f1["wall_s"], f5["wall_s"]
        out.append(f"- final metrics identical: {same_final}; include bits identical: {same_bits}; rules identical: "
                   f"{rules(e1) == rules(e5)}; wall {w1:.0f} s vs {w5:.0f} s")
        verdict["eval_every_5"] = same_final and same_bits
        sl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "share_test", "saveload_check.log")
        if os.path.exists(sl):
            sc = [json.loads(l.split(" ", 1)[1]) for l in open(sl) if l.startswith("SCORE ")]
            if sc:
                same = sc[0]["auc"] == f1["dev_auc"] and sc[0]["acc"] == f1["dev_acc"] and sc[0]["pred_pos"] == f1["pred_pos"]
                out.append(f"- save/load: reloaded dev AUC {sc[0]['auc']:.6f} / acc {sc[0]['acc']} vs in-run {f1['dev_auc']:.6f} / {f1['dev_acc']} -> identical: {same}")
                verdict["save_load_exact"] = same
            else:
                out.append("- save/load: no SCORE line (see share_test/saveload_check.log)")
    else:
        out.append("- not run or missing")
    out.append("")
    out.append("VERDICT " + json.dumps({f"{k[0]}:{k[1]}" if isinstance(k, tuple) else k: v for k, v in verdict.items()}))
    print("\n".join(out))


if __name__ == "__main__":
    main()
