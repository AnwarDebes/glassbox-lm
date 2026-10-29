"""REG_agent_vs_llm: the one table of every rule's outcome, built from the three report files and nothing else.

usage: python3.12 -I reg_table.py [--dir runs/reg] [--out runs/reg/table.md]
Reads DIR/report_<ds>.json (reg_report.py), and when present DIR/checks.json (check_reg.py's gate) and
DIR/verify_<ds>.out (verify_reg.py's verdict). No statistic is computed here: the numbers are the reports'. The one
reading added is the one the registration names (section 3): beside every rule met on the registered seed-1 interval,
whether the 5-seed two-level interval (two_level.ci, two_level.ft_minus_agent_ci) meets the same bound, judged with
the 5-seed-mean difference (two_level.agent_minus_llm_mean) for SUP's +3.0 part. Standard library only.
"""
import argparse
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("star", "abcd", "multiwoz")
NAMES = {"star": "STAR", "abcd": "ABCD", "multiwoz": "MultiWOZ"}


def pts(x):
    return f"{100 * x:+.2f}"


def ci(c):
    return f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"


def five_seed_meets(rep, rule):
    """Does the two-level (seeds 1-5, then dialogues) reading meet the rule's bound?"""
    tl = rep.get("two_level") or {}
    lo = (tl.get("ci") or [None])[0]
    if rule == "FT":
        lo_ft = (tl.get("ft_minus_agent_ci") or [None])[0]
        return None if lo_ft is None else lo_ft > 0
    if lo is None:
        return None
    if rule == "NI":
        return lo > -0.03
    if rule == "NI2":
        return lo > -0.02
    if rule == "SUP":
        return tl.get("agent_minus_llm_mean", -1) >= 0.03 and lo > 0
    raise ValueError(rule)


def verdict(path):
    if not os.path.exists(path):
        return "not run"
    try:
        return json.load(open(path)).get("verdict", "unreadable")
    except (json.JSONDecodeError, OSError):
        return "unreadable"


def table(d):
    gates = (json.load(open(os.path.join(d, "checks.json"))).get("gate", {})
             if os.path.exists(os.path.join(d, "checks.json")) else {})
    rows = ["| Dataset | Rows | AGENT | LLM* | FT | AGENT - LLM* (95%) | NI (-3) | NI2 (-2) | SUP (+3) | FT - AGENT (95%) "
            "| FT | 5-seed mean AGENT - LLM* (two-level 95%) | Verify |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    claims, notes, reps = [], [], {}
    for ds in DATASETS:
        p = os.path.join(d, f"report_{ds}.json")
        if gates and not gates.get(ds, False):
            rows.append(f"| {NAMES[ds]} | gate closed: not completed (check_reg.py) | | | | | | | | | | | |")
            continue
        if not os.path.exists(p):
            rows.append(f"| {NAMES[ds]} | no report | | | | | | | | | | | |")
            continue
        r = json.load(open(p))
        reps[ds] = r
        tl = r.get("two_level") or {}

        def mark(rule, met):
            if not met:
                return "not met"
            five = five_seed_meets(r, rule)
            if five is False:
                notes.append(f"{NAMES[ds]} {rule}: met on seed 1, not on the 5-seed mean")
                return "met (seed 1 only)"
            return "met" if five else "met (5-seed n/a)"

        rows.append(
            f"| {NAMES[ds]} | {r['n_rows']} | {r['acc_agent']:.4f} | {r['acc_llm']:.4f} | {r['acc_ft']:.4f} "
            f"| {pts(r['agent_minus_llm'])} {ci(r['ci_agent_minus_llm'])} | {mark('NI', r['NI_met'])} "
            f"| {mark('NI2', r['NI2_met'])} | {mark('SUP', r['SUP_met'])} "
            f"| {pts(r['ft_minus_agent'])} {ci(r['ci_ft_minus_agent'])} | {mark('FT', r['FT_met'])} "
            f"| {pts(tl['agent_minus_llm_mean'])} {ci(tl['ci'])} | {verdict(os.path.join(d, f'verify_{ds}.out'))} |"
            if tl else
            f"| {NAMES[ds]} | {r['n_rows']} | {r['acc_agent']:.4f} | {r['acc_llm']:.4f} | {r['acc_ft']:.4f} "
            f"| {pts(r['agent_minus_llm'])} {ci(r['ci_agent_minus_llm'])} | {mark('NI', r['NI_met'])} "
            f"| {mark('NI2', r['NI2_met'])} | {mark('SUP', r['SUP_met'])} "
            f"| {pts(r['ft_minus_agent'])} {ci(r['ci_ft_minus_agent'])} | {mark('FT', r['FT_met'])} | n/a "
            f"| {verdict(os.path.join(d, f'verify_{ds}.out'))} |")
    nr = ["| Dataset | AGENT - LLM* 90% (inside +-3?) | TFIDF | TFIDF - LLM* (95%) | AGENT - TFIDF (95%) "
          "| AGENT - TFIDF two-level | FT - AGENT two-level | LLM* s/row (median) | LLM* invalid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, r in reps.items():
        t = r.get("tfidf_no_rule") or {}
        tl = r.get("two_level") or {}
        nr.append(f"| {NAMES[ds]} | {ci(r['ci90_agent_minus_llm'])} ({'yes' if r['EQ_inside_no_rule'] else 'no'}) "
                  f"| {t.get('acc_tfidf', float('nan')):.4f} | {pts(t['tfidf_minus_llm'])} {ci(t['ci_tfidf_minus_llm'])} "
                  f"| {pts(t['agent_minus_tfidf'])} {ci(t['ci_agent_minus_tfidf'])} "
                  f"| {ci(tl['agent_minus_tfidf_ci']) if tl.get('agent_minus_tfidf_ci') else 'n/a'} "
                  f"| {ci(tl['ft_minus_agent_ci']) if tl.get('ft_minus_agent_ci') else 'n/a'} "
                  f"| {r.get('llm_median_s_per_row')} | {r.get('llm_invalid')} |")
    ni = [ds for ds, r in reps.items() if r["NI_met"]]
    ni2 = [ds for ds, r in reps.items() if r["NI2_met"]]
    if len(ni) == 3:
        margin = "2 points (NI2)" if len(ni2) == 3 else "3 points (NI)"
        claims.append(f"NI met on all three datasets: the readable agent is not worse than the best of the local LLM "
                      f"configurations we ran (chosen on dev; the 35B model at 4-bit), at a margin of {margin}.")
        if len(ni2) < 3:
            claims.append("NI2 (2 points) met on: " + (", ".join(NAMES[x] for x in ni2) or "none") + "; not on: "
                          + ", ".join(NAMES[x] for x in reps if x not in ni2) + ".")
    else:
        for ds, r in reps.items():
            claims.append(f"{NAMES[ds]}: NI {'met' if r['NI_met'] else 'not met'}, NI2 "
                          f"{'met' if r['NI2_met'] else 'not met'} (AGENT - LLM* {pts(r['agent_minus_llm'])} "
                          f"{ci(r['ci_agent_minus_llm'])}).")
    for ds, r in reps.items():
        if r["SUP_met"]:
            claims.append(f"{NAMES[ds]}: SUP met (AGENT - LLM* {pts(r['agent_minus_llm'])} {ci(r['ci_agent_minus_llm'])}).")
    return "\n".join(rows), "\n".join(nr), claims, notes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(H, "runs/reg"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rules, norule, claims, notes = table(a.dir)
    text = ("## Every rule's outcome (registered statistic: seed 1, bootstrap over dialogues, 95%)\n\n" + rules
            + "\n\n## No-rule readings\n\n" + norule + "\n\n## Claim lines\n\n" + "\n".join(f"- {c}" for c in claims)
            + ("\n\n## Seed-1-only notes\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "") + "\n")
    print(text)
    if a.out:
        with open(a.out, "x") as h:
            h.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
