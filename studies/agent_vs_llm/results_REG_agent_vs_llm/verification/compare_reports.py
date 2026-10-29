"""Compare reg_report.py's output with recompute_reg.py's (same rows and inputs, independent bootstrap).

usage: python -I compare_reports.py REG_REPORT.json RECOMPUTE.json [--tol 0.005]
Prints one line per number; exit 0 when every point estimate is equal (1e-12), every interval end is within --tol
(fraction; 0.005 = 0.5 points) and every rule agrees, except rules whose registered lower end lies within 0.003 of
its bound (NEAR-BOUND: the registered interval decides). Exit 1 otherwise.
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reg")
    ap.add_argument("mine")
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args(argv)
    r, m = json.load(open(a.reg)), json.load(open(a.mine))
    bad = []

    def point(name, x, y):
        ok = abs(x - y) <= 1e-12
        print(f"{'ok ' if ok else 'BAD'} point {name}: registered {x:.6f} recompute {y:.6f}")
        if not ok:
            bad.append(name)

    def interval(name, x, y):
        dif = max(abs(x[0] - y[0]), abs(x[1] - y[1]))
        ok = dif <= a.tol
        print(f"{'ok ' if ok else 'BAD'} ci {name}: registered [{100 * x[0]:+.2f}, {100 * x[1]:+.2f}] recompute "
              f"[{100 * y[0]:+.2f}, {100 * y[1]:+.2f}] points; max end difference {100 * dif:.2f}")
        if not ok:
            bad.append(name)

    for k in ("n_rows", "n_dialogues", "llm_invalid"):
        ok = r[k] == m[k]
        print(f"{'ok ' if ok else 'BAD'} {k}: {r[k]} / {m[k]}")
        if not ok:
            bad.append(k)
    for k in ("agent", "llm", "ft"):
        point(f"acc_{k}", r[f"acc_{k}"], m["acc"][k])
    point("agent_minus_llm", r["agent_minus_llm"], m["agent_minus_llm"]["mean"])
    point("ft_minus_agent", r["ft_minus_agent"], m["ft_minus_agent"]["mean"])
    interval("agent_minus_llm 95", r["ci_agent_minus_llm"], m["agent_minus_llm"]["ci95"])
    interval("agent_minus_llm 90", r["ci90_agent_minus_llm"], m["agent_minus_llm"]["ci90"])
    interval("ft_minus_agent 95", r["ci_ft_minus_agent"], m["ft_minus_agent"]["ci95"])
    t = r.get("tfidf_no_rule")
    if t and "tfidf" in m["acc"]:
        point("acc_tfidf", t["acc_tfidf"], m["acc"]["tfidf"])
        point("tfidf_minus_llm", t["tfidf_minus_llm"], m["tfidf_minus_llm"]["mean"])
        point("agent_minus_tfidf", t["agent_minus_tfidf"], m["agent_minus_tfidf"]["mean"])
        interval("tfidf_minus_llm 95", t["ci_tfidf_minus_llm"], m["tfidf_minus_llm"]["ci95"])
        interval("agent_minus_tfidf 95", t["ci_agent_minus_tfidf"], m["agent_minus_tfidf"]["ci95"])
    bounds = {"NI_met": ("ci_agent_minus_llm", 0, -0.03), "NI2_met": ("ci_agent_minus_llm", 0, -0.02),
              "SUP_met": ("ci_agent_minus_llm", 0, 0.0), "FT_met": ("ci_ft_minus_agent", 0, 0.0),
              "EQ_inside_no_rule": ("ci90_agent_minus_llm", 0, -0.03)}
    for k, (ck, end, bound) in bounds.items():
        near = abs(r[ck][end] - bound) <= 0.003
        same = r[k] == m["rules"][k]
        tag = "ok " if same else ("NEAR-BOUND" if near else "BAD")
        print(f"{tag} rule {k}: registered {r[k]} recompute {m['rules'][k]}"
              + (f" (registered end {100 * r[ck][end]:+.2f} within 0.3 points of {100 * bound:+.1f})" if near else ""))
        if not (same or near):
            bad.append(k)
    tl, ml = r.get("two_level"), m.get("two_level")
    if tl and ml:
        point("two_level agent_minus_llm_mean", tl["agent_minus_llm_mean"], ml["agent_minus_llm_mean"])
        for i, (x, y) in enumerate(zip(tl["agent_acc_per_seed"], ml["agent_acc_per_seed"])):
            point(f"two_level seed {i + 1} acc", x, y)
        interval("two_level agent_minus_llm 95", tl["ci"], ml["ci_agent_minus_llm"])
        interval("two_level ft_minus_agent 95", tl["ft_minus_agent_ci"], ml["ci_ft_minus_agent"])
        if "agent_minus_tfidf_ci" in tl and "ci_agent_minus_tfidf" in ml:
            interval("two_level agent_minus_tfidf 95", tl["agent_minus_tfidf_ci"], ml["ci_agent_minus_tfidf"])
    print("MATCH" if not bad else f"MISMATCH: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
