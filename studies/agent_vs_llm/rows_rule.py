"""REG_agent_vs_llm: the number of registered test rows per dataset, from dev only (rule fixed in draft v8).

usage: python -I rows_rule.py --choice runs/llm_star_choice.json --out runs/reg_rows.json
Rule: per dataset, d = AGENT minus LLM* per row on LLM*'s 500 dev rows (agent seed 1's dev predictions); with
se(n) = sd(d) * 1.1 / sqrt(n) (1.1 for dialogue clustering) and the planning difference D = min(mean(d), 0) (no
credit for a dev lead; a dev deficit is kept), NI power(n) = Phi((D + 0.03) / se(n) - 1.96). A test split of at most
4,000 rows is used whole (ABCD: 3,608; it costs the same test use). Otherwise the rows are the smallest n in 2,000,
3,000, 4,000 with power >= 0.90; if none reaches it, 4,000, and the power is reported. Test split sizes are taken from
earlier registered reports (R19 STAR and ABCD, R24 MultiWOZ: n_test_action_turns); no test file is read.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import json
import math
import os
import struct
import zipfile

H = os.path.dirname(os.path.abspath(__file__))
N_TEST = {"star": 6988, "abcd": 3608, "multiwoz": 7355}
AGENT_DEV = {"star": "runs/agent_star_c4_tm_s1/tm.npz", "abcd": "runs/agent_abcd_c4_tm_s1/tm.npz",
             "multiwoz": "runs/agent_multiwoz_c4_tm_s1/tm.npz"}
DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
AGENT_DEV_ACC = {"star": 0.6544585987261147, "abcd": 0.8417480998914224, "multiwoz": 0.35004756080989263}
MARGIN, INFLATE, TARGET, STEPS = 0.03, 1.1, 0.90, (2000, 3000, 4000)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def power(mean, sd, n):
    se = sd * INFLATE / math.sqrt(n)
    return phi((mean + MARGIN) / se - 1.96)


def choose(mean, sd, n_test):
    plan = min(mean, 0.0)
    if n_test <= max(STEPS):
        return n_test, power(plan, sd, n_test)
    cands = sorted(STEPS)
    for n in cands:
        if power(plan, sd, n) >= TARGET:
            return n, power(plan, sd, n)
    return cands[-1], power(plan, sd, cands[-1])


def top1(path):
    """Column 0 of top_ids and gold from an .npz (minimal .npy reader: little-endian int arrays, C order)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for key in ("top_ids", "gold"):
            raw = z.read(key + ".npy")
            hl = struct.unpack("<H", raw[8:10])[0] if raw[6] == 1 else struct.unpack("<I", raw[8:12])[0]
            st = 10 if raw[6] == 1 else 12
            hd = eval(raw[st:st + hl].decode("latin1"), {"__builtins__": {}}, {"True": True, "False": False})
            code = {"<i2": "h", "<i4": "i", "<i8": "q"}[hd["descr"]]
            assert not hd["fortran_order"]
            n = 1
            for s in hd["shape"]:
                n *= s
            v = struct.unpack("<" + code * n, raw[st + hl:st + hl + n * struct.calcsize(code)])
            cols = hd["shape"][1] if len(hd["shape"]) == 2 else 1
            out[key] = [v[i * cols] for i in range(hd["shape"][0])]
    return out["top_ids"], out["gold"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = json.load(open(a.choice))["datasets"]
    res = {"rule": __doc__.split("Rule: ")[1].split(" Test split")[0].replace("\n", " "), "datasets": {}}
    for ds in ("star", "abcd", "multiwoz"):
        star = ch[ds]["llm_star"]
        with gzip.open(f"{DATA}/{ds}/examples_dev.json.gz", "rt") as h:
            d = json.load(h)
        nid = {b: i for i, b in enumerate(d["buttons"])}
        gold = [r["y"] for r in d["rows"]]
        top, g2 = top1(os.path.join(H, AGENT_DEV[ds]))
        assert g2 == gold, ds
        acc = sum(int(x == y) for x, y in zip(top, gold)) / len(gold)
        assert abs(acc - AGENT_DEV_ACC[ds]) < 1e-12, (ds, acc)  # the agent's registered dev run
        recs = [json.loads(x) for x in open(os.path.join(H, star["path"], "preds.jsonl")) if x.strip()]
        assert len(recs) == 500, (ds, len(recs))
        diff = [float(top[r["i"]] == gold[r["i"]]) - float(nid.get(r["pred"], -1) == gold[r["i"]]) for r in recs]
        m = sum(diff) / len(diff)
        sd = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1))
        n, pw = choose(m, sd, N_TEST[ds])
        res["datasets"][ds] = {"llm_star": star["config"], "dev_rows": len(diff), "dev_agent_minus_llm": m, "sd": sd,
                               "planning_difference": min(m, 0.0), "n_test_split": N_TEST[ds], "rows": n,
                               "power_at_planning_difference": pw, "power_at_dev_estimate": power(m, sd, n),
                               "power_at_zero": power(0.0, sd, n), "target_met": pw >= TARGET}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    for ds, v in res["datasets"].items():
        print(f"{ds:8s} LLM* {v['llm_star']:8s} dev diff {v['dev_agent_minus_llm']:+.3f} sd {v['sd']:.3f} -> rows "
              f"{v['rows']} (power {v['power_at_planning_difference']:.2f} at the planning difference "
              f"{v['planning_difference']:+.3f}; {v['power_at_dev_estimate']:.2f} at the dev estimate)")


if __name__ == "__main__":
    main()
