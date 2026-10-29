"""PLAN_logsize_dev.md: the log-size table (dev only).

usage (decision_layer's .venv-dl): python -I logsize_table.py [--root .] [--out runs/logsize/table] [--draws 10000]
Reads the agent's tm.npz, the LLM's preds.jsonl and TFIDF's json for every dataset and size (the whole split from the
earlier runs named in the plan), checks that every LLM run answered L35k50's 500 rows, and writes OUT.md and OUT.json
(never overwritten). Statistic and readings as fixed in the plan: accuracy on the 500 rows, paired differences with a
95% bootstrap interval over rows (seed 20261026), the label per size, N_reach per dataset.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import json
import os
import statistics
import sys

H = os.path.dirname(os.path.abspath(__file__))
DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
DATASETS = ("star", "abcd", "multiwoz")
NAMES = {"star": "STAR", "abcd": "ABCD", "multiwoz": "MultiWOZ"}
SIZES = ("64", "256", "1024", "4096", "whole")
SEED = 20261026
AGENT_WHOLE = {"star": "runs/agent_star_c4_tm_s1/tm.npz", "abcd": "runs/agent_abcd_c4_tm_s1/tm.npz",
               "multiwoz": "runs/logsize/agent_multiwoz_whole/tm.npz"}
E84_MWOZ_DEV = 0.35004756080989263  # e84 / R24 seed 1, the earlier whole-split MultiWOZ agent (written beside)


def paths(root, ds, size):
    agent = os.path.join(root, AGENT_WHOLE[ds] if size == "whole" else f"runs/logsize/agent_{ds}_N{size}/tm.npz")
    llm = os.path.join(root, f"runs/llm_q3_L35k50/{ds}/preds.jsonl" if size == "whole"
                       else f"runs/logsize/llm_N{size}/{ds}/preds.jsonl")
    tfidf = os.path.join(root, f"runs/logsize/tfidf/{ds}_{size}.json")
    return agent, llm, tfidf


def boot(diff, draws, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    b = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def label(ci):
    if ci[1] < 0:
        return "LLM* ahead"
    if ci[0] > 0:
        return "AGENT ahead"
    return "level within the interval"


def n_reach(diffs):
    """diffs: [(size, AGENT - LLM* or None)] in size order; the smallest size from which on every diff >= 0."""
    out = None
    for size, d in reversed(diffs):
        if d is None or d < 0:
            break
        out = size
    return out or "not reached"


def gate(root):
    p = os.path.join(root, "runs/logsize/agent_abcd_whole/result.json")
    if not os.path.exists(p):
        return False, "gate not run"
    chk = json.load(open(p)).get("dev_check") or {}
    return bool(chk.get("equal")), f"ABCD whole-split rerun {chk.get('got')} vs {chk.get('expected')}"


def table(root, draws):
    import numpy as np
    gate_ok, gate_note = gate(root)
    out = {"plan": "PLAN_logsize_dev.md", "gate_ok": gate_ok, "gate": gate_note, "datasets": {}}
    for ds in DATASETS:
        dv = json.load(gzip.open(os.path.join(DATA, ds, "examples_dev.json.gz"), "rt"))
        buttons, gold_all = dv["buttons"], np.array([int(r["y"]) for r in dv["rows"]])
        nid = {b: i for i, b in enumerate(buttons)}
        base = [json.loads(x)["i"] for x in open(os.path.join(root, f"runs/llm_q3_L35k50/{ds}/preds.jsonl")) if x.strip()]
        idx = np.array(sorted(base))
        gold = gold_all[idx]
        rows, diffs = [], []
        for size in SIZES:
            pa, pl, pt = paths(root, ds, size)
            r = {"size": size}
            ok = {}
            if os.path.exists(pt):
                t = json.load(open(pt))
                r.update(n_dialogues=t["n_dialogues"], n_train_rows=t["n_train_rows"],
                         unseen_gold_500=t["unseen_gold_500"])
                if size == "whole" and not t.get("equal_to_dry_run", False):
                    r["tfidf_note"] = "not read: differs from tfidf_reg.py's dry run"
                else:
                    ok["TFIDF"] = (np.array(t["top1_dev"])[idx] == gold).astype(float)
            if os.path.exists(pa) and (gate_ok or size == "whole" and ds != "multiwoz"):
                A = np.load(pa)
                assert np.array_equal(A["gold"].astype(int), gold_all), f"{pa}: gold is not dev's"
                ok["AGENT"] = (A["top_ids"][:, 0].astype(int)[idx] == gold).astype(float)
            elif os.path.exists(pa):
                r["agent_note"] = "not read: " + gate_note
            if os.path.exists(pl):
                recs = {json.loads(x)["i"]: json.loads(x) for x in open(pl) if x.strip()}
                if sorted(recs) == sorted(base):
                    pred = np.array([nid.get(recs[i]["pred"], -1) for i in idx])
                    ok["LLM*"] = (pred == gold).astype(float)
                    r["llm_invalid"] = int(sum(recs[i]["pred"] is None for i in idx))
                    r["llm_errors"] = int(sum(recs[i].get("error") is not None for i in idx))
                    r["llm_median_s"] = statistics.median(recs[i]["s"] for i in idx)
                else:
                    r["llm_note"] = f"incomplete or other rows: {len(recs)} answered"
            for arm, v in ok.items():
                r[f"acc_{arm}"] = float(v.mean())
            for x, y in (("AGENT", "LLM*"), ("TFIDF", "LLM*"), ("AGENT", "TFIDF")):
                if x in ok and y in ok:
                    d = ok[x] - ok[y]
                    r[f"{x} - {y}"] = float(d.mean())
                    r[f"{x} - {y} ci95"] = boot(d, draws)
            if "AGENT - LLM* ci95" in r:
                r["label"] = label(r["AGENT - LLM* ci95"])
            diffs.append((size, r.get("AGENT - LLM*")))
            rows.append(r)
        res = {"n_rows": int(len(idx)), "rows": rows, "N_reach": n_reach(diffs)}
        if ds == "multiwoz":
            res["e84_whole_dev_acc_all_rows"] = E84_MWOZ_DEV
            pr = os.path.join(root, "runs/logsize/agent_multiwoz_whole/result.json")
            if os.path.exists(pr):
                res["rerun_whole_dev_acc_all_rows"] = json.load(open(pr))["arms"]["tm"]["dev_acc"]
        out["datasets"][ds] = res
    return out


def pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def ci(r, k):
    if k not in r:
        return ""
    return f"{100 * r[k]:+.1f} [{100 * r[k + ' ci95'][0]:+.1f}, {100 * r[k + ' ci95'][1]:+.1f}]"


def markdown(out):
    lines = ["# Log-size table (PLAN_logsize_dev.md; dev, 500 rows per dataset; 95% bootstrap over rows)", "",
             f"Gate (ABCD whole-split rerun equal to the earlier run): {'passed' if out['gate_ok'] else 'NOT passed'}"
             f" ({out['gate']}).", ""]
    for ds, res in out["datasets"].items():
        lines += [f"## {NAMES[ds]} (N_reach: {res['N_reach']})", "",
                  "| Dialogues | Train rows | Gold unseen in pool | AGENT | LLM* | TFIDF | AGENT - LLM* | Label "
                  "| TFIDF - LLM* | AGENT - TFIDF | LLM* invalid | LLM* s/row |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["rows"]:
            lines.append(f"| {r.get('n_dialogues', r['size'])} | {r.get('n_train_rows', '')} "
                         f"| {pct(r.get('unseen_gold_500'))} | {pct(r.get('acc_AGENT'))} | {pct(r.get('acc_LLM*'))} "
                         f"| {pct(r.get('acc_TFIDF'))} | {ci(r, 'AGENT - LLM*')} | {r.get('label', '')} "
                         f"| {ci(r, 'TFIDF - LLM*')} | {ci(r, 'AGENT - TFIDF')} | {r.get('llm_invalid', '')} "
                         f"| {r.get('llm_median_s', '')} |")
        notes = [f"{r['size']}: {r[k]}" for r in res["rows"] for k in ("agent_note", "llm_note", "tfidf_note") if k in r]
        if ds == "multiwoz" and "rerun_whole_dev_acc_all_rows" in res:
            notes.append(f"whole-split agent rerun, all dev rows: {res['rerun_whole_dev_acc_all_rows']:.8f} "
                         f"(e84: {res['e84_whole_dev_acc_all_rows']:.8f})")
        lines += [""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=H)
    ap.add_argument("--out", default=os.path.join(H, "runs/logsize/table"))
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args(argv)
    out = table(a.root, a.draws)
    md = markdown(out)
    print(md)
    with open(a.out + ".json", "x") as h:
        json.dump(out, h, indent=1)
    with open(a.out + ".md", "x") as h:
        h.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
