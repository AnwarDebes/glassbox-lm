"""Independent recompute of REG_agent_vs_llm (verify-result): own code, standard library only, another bootstrap seed.

It does not import reg_report.py. From the raw files it recomputes, per dataset: the rows the LLM* run answered (and,
on test, that they are the registered draw random.Random(20261027).sample(range(n_test), n) with n from
runs/reg_rows.json, with no request error), each arm's accuracy on them, AGENT minus LLM* and FT minus AGENT, and
percentile intervals from a bootstrap over dialogues with seed 7; then it compares with the report file and prints
MATCH or MISMATCH per item. The registered interval (reg_report.py) decides each rule. Where the report's lower end
lies within 0.003 of a rule's bound and this recompute's interval falls on the other side, the item is marked
NEAR-BOUND (bootstrap noise between two seeds), not MISMATCH, and is reported.
usage: python3.12 -I verify_reg.py --dataset star --split test --n ROWS --reg REG_agent_vs_llm.md \
           --report runs/reg/report_star.json --agent <seed1.npz> --llm runs/reg/llm_star/preds.jsonl \
           --ft runs/reg/ft_star/test_preds.npz [--tfidf runs/reg/tfidf_star/test_preds.npz]
(.npz files are read with a minimal reader of numpy's .npy format, so numpy is not needed; both file formats: pred + y
or top_ids + gold.)
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import hashlib
import json
import os
import random
import struct
import sys
import zipfile

DATA = f"{_REPO_ROOT}/studies/decision_layer/data"


def read_npy(raw):
    """Minimal .npy reader for little-endian int16/int32/int64 arrays of 1 or 2 dimensions (C order)."""
    assert raw[:6] == b"\x93NUMPY"
    major = raw[6]
    hlen = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
    start = 10 if major == 1 else 12
    header = raw[start:start + hlen].decode("latin1")
    hd = eval(header, {"__builtins__": {}}, {"True": True, "False": False})  # a dict literal written by numpy
    code = {"<i2": "h", "<i4": "i", "<i8": "q"}[hd["descr"]]
    assert not hd["fortran_order"]
    shape = hd["shape"]
    n = 1
    for s in shape:
        n *= s
    data = struct.unpack("<" + code * n, raw[start + hlen:start + hlen + n * struct.calcsize(code)])
    if len(shape) == 1:
        return list(data)
    cols = shape[1]
    return [list(data[i * cols:(i + 1) * cols]) for i in range(shape[0])]


def decisions(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "pred.npy" in names:
            return read_npy(z.read("pred.npy")), read_npy(z.read("y.npy"))
        return [r[0] for r in read_npy(z.read("top_ids.npy"))], read_npy(z.read("gold.npy"))


def sup_agrees(diff, lo7, lo_rep, sup_met_rep):
    """Does the seed-7 recompute agree with the report on SUP (difference >= +0.03 and lower end > 0)?

    Returns (agree, note). A disagreement counts as NEAR-BOUND, not MISMATCH, when the report's lower end is within
    0.003 of 0, or when the difference is +0.03 up to rounding (mean(a) - mean(b) here, mean(a - b) in reg_report.py).
    """
    if (diff >= 0.03 and lo7 > 0) == sup_met_rep:
        return True, None
    if abs(lo_rep) < 0.003:
        return True, f"SUP: report lower end {lo_rep:+.4f}, seed-7 lower end {lo7:+.4f}, bound 0"
    if abs(diff - 0.03) < 1e-9:
        return True, f"SUP: difference {diff:+.12f} is +0.03 up to rounding"
    return False, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("dataset", "split", "report", "agent", "llm", "ft"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--reg", default=None)
    ap.add_argument("--tfidf", default=None, help="TFIDF arm (no rule)")
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args(argv)
    if a.split == "test":
        assert a.n and a.reg, "--n and --reg are required on the test split"
        fr = os.path.splitext(os.path.abspath(a.reg))[0] + ".frozen.json"
        if not os.path.exists(fr):
            raise SystemExit(f"refused: no freeze record {fr}")
        if hashlib.sha256(open(a.reg, "rb").read()).hexdigest() != json.load(open(fr))["sha256"]:
            raise SystemExit("refused: the registration differs from its freeze record")
    with gzip.open(f"{DATA}/{a.dataset}/examples_{a.split}.json.gz", "rt") as h:
        d = json.load(h)
    rows, buttons = d["rows"], d["buttons"]
    nid = {b: i for i, b in enumerate(buttons)}
    gold = [r["y"] for r in rows]
    recs = [json.loads(x) for x in open(a.llm) if x.strip()]
    idx = sorted(r["i"] for r in recs)
    want = sorted(random.Random(20261027).sample(range(len(rows)), min(a.n, len(rows)))) if a.n else None
    llm = {r["i"]: nid.get(r["pred"], -1) for r in recs}
    n_err = sum(r.get("error") is not None for r in recs)
    agent_dec, g1 = decisions(a.agent)
    ft_dec, g2 = decisions(a.ft)
    assert g1 == gold and g2 == gold
    ag = [1.0 if agent_dec[i] == gold[i] else 0.0 for i in idx]
    ll = [1.0 if llm[i] == gold[i] else 0.0 for i in idx]
    ft = [1.0 if ft_dec[i] == gold[i] else 0.0 for i in idx]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    by = {}
    for j, i in enumerate(idx):
        by.setdefault(rows[i]["convo_id"], []).append(j)
    groups = list(by.values())
    rng = random.Random(7)

    def boot(x, y, lo=0.025):
        out = []
        for _ in range(a.draws):
            pick = [groups[rng.randrange(len(groups))] for _ in groups]
            rr = [j for g in pick for j in g]
            out.append(sum(x[j] - y[j] for j in rr) / len(rr))
        out.sort()
        return out[int(lo * len(out))], out[int((1 - lo) * len(out)) - 1]

    res = {"rows_are_registered_draw": idx == want if a.split == "test" else None, "llm_errors": n_err,
           "n": len(idx), "acc_agent": mean(ag), "acc_llm": mean(ll), "acc_ft": mean(ft),
           "agent_minus_llm": mean(ag) - mean(ll), "ci_agent_minus_llm_seed7": boot(ag, ll),
           "ft_minus_agent": mean(ft) - mean(ag), "ci_ft_minus_agent_seed7": boot(ft, ag),
           "ci90_agent_minus_llm_seed7": boot(ag, ll, 0.05)}
    if a.tfidf:
        tf_dec, g3 = decisions(a.tfidf)
        assert g3 == gold
        tf = [1.0 if tf_dec[i] == gold[i] else 0.0 for i in idx]
        res["acc_tfidf"] = mean(tf)
        res["ci_tfidf_minus_llm_seed7"] = boot(tf, ll)
        res["ci_agent_minus_tfidf_seed7"] = boot(ag, tf)
    rep = json.load(open(a.report))
    checks, near = {}, []
    for k in ("acc_agent", "acc_llm", "acc_ft", "agent_minus_llm", "ft_minus_agent"):
        checks[k] = abs(rep[k] - res[k]) < 1e-9
    for k, rk in (("ci_agent_minus_llm_seed7", "ci_agent_minus_llm"), ("ci_ft_minus_agent_seed7", "ci_ft_minus_agent"),
                  ("ci90_agent_minus_llm_seed7", "ci90_agent_minus_llm")):
        checks[k + " within 0.01 of the report"] = all(abs(x - y) <= 0.01 for x, y in zip(res[k], rep[rk]))
    if a.tfidf:
        tr = rep["tfidf_no_rule"]
        checks["acc_tfidf"] = abs(tr["acc_tfidf"] - res["acc_tfidf"]) < 1e-9
        checks["ci_tfidf_minus_llm within 0.01"] = all(
            abs(x - y) <= 0.01 for x, y in zip(res["ci_tfidf_minus_llm_seed7"], tr["ci_tfidf_minus_llm"]))
        checks["ci_agent_minus_tfidf within 0.01"] = all(
            abs(x - y) <= 0.01 for x, y in zip(res["ci_agent_minus_tfidf_seed7"], tr["ci_agent_minus_tfidf"]))
    checks["margins are 0.03 and 0.02"] = rep["margin"] == 0.03 and rep["margin2"] == 0.02
    checks["no LLM request error"] = n_err == 0 and rep["llm_errors"] == 0
    lo7 = res["ci_agent_minus_llm_seed7"][0]
    lo_rep = rep["ci_agent_minus_llm"][0]
    for name, bound, key in (("NI", -0.03, "NI_met"), ("NI2", -0.02, "NI2_met")):
        agree = (lo7 > bound) == rep[key]
        if not agree and abs(lo_rep - bound) < 0.003:
            near.append(f"{name}: report lower end {lo_rep:+.4f}, seed-7 lower end {lo7:+.4f}, bound {bound:+.2f}")
            agree = True
        checks[f"rule {name} agrees"] = agree
    sup_agree, note = sup_agrees(res["agent_minus_llm"], lo7, lo_rep, rep["SUP_met"])
    if note:
        near.append(note)
    checks["rule SUP agrees"] = sup_agree
    lo_ft7, lo_ft = res["ci_ft_minus_agent_seed7"][0], rep["ci_ft_minus_agent"][0]
    ft_agree = (lo_ft7 > 0) == rep["FT_met"]
    if not ft_agree and abs(lo_ft) < 0.003:
        near.append(f"FT: report lower end {lo_ft:+.4f}, seed-7 lower end {lo_ft7:+.4f}, bound 0")
        ft_agree = True
    checks["rule FT agrees"] = ft_agree
    lo90, hi90 = res["ci90_agent_minus_llm_seed7"]
    eq7 = lo90 > -0.03 and hi90 < 0.03
    checks["equivalence reading agrees"] = eq7 == rep["EQ_inside_no_rule"] or min(
        abs(rep["ci90_agent_minus_llm"][0] + 0.03), abs(rep["ci90_agent_minus_llm"][1] - 0.03)) < 0.003
    res["checks"] = checks
    res["near_bound"] = near
    ok_rows = res["rows_are_registered_draw"] in (True, None)
    res["verdict"] = "MATCH" if all(checks.values()) and ok_rows else "MISMATCH"
    print(json.dumps(res, indent=1))
    return 0 if res["verdict"] == "MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
