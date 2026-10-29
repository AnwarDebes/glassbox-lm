"""REG_agent_vs_llm: the checks that must pass after the test runs and before reg_report.py (standard library only).

usage: python3.12 -I check_reg.py --reg REG_agent_vs_llm.md --rows runs/reg_rows.json --out runs/reg/checks.json \
           [--choice runs/llm_star_choice.json] [--jobs jobs_reg.txt] [--queue-log queue.log]
Hard checks, each one either global or of one dataset:
  - global: the freeze record exists; every pin of the freeze record is unchanged (freeze_reg.check);
  - AGENT: the registered seed files are those named in AGENT_FILES; seed 1's recorded dev accuracy (seed1.json
    dev_acc) equals the reference exactly;
  - LLM*: per dataset, its test preds.jsonl holds exactly the registered draw and no request error, and result.json
    exists with n equal to the rows; its argv and result.json hold LLM*'s registered settings (k, reasoning, policy
    text, max tokens, the row seed and n; runs/llm_star_choice.json);
  - FT and TFIDF: their test prediction files and result.json exist and are not dry runs; FT ran bge-base, lr 5e-5,
    seed 1, 4 epochs, batch 32, 256 tokens; TFIDF ran the registered C;
  - the queue ran each of the dataset's three job lines exactly as pinned in jobs_reg.txt (every "start:" line of
    queue.log for reg_ft_<ds>, reg_tfidf_<ds> and reg_llm_<ds>).
Gate: a dataset is reported only when the global checks and all of its own checks pass (out["gate"]); a failure on one
dataset does not stop the others. Exit 0 when every dataset's gate is open, else 1.
Flags (reported, not fatal): FT's dev accuracy more than 0.5 points from the earlier dev run; TFIDF's dev accuracy
more than 0.5 points from tfidf_lr.py's; LLM*'s invalid answers above 0.5% of its rows (then the server log is read).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import random
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import freeze_reg  # noqa: E402

CONFIRM = f"{_REPO_ROOT}/studies/decision_layer/results/confirm"
AGENT_DIR = {"star": f"{CONFIRM}/R22star/tm_mem_c4_s16", "abcd": f"{CONFIRM}/R23abcd/tm_mem_c4_s16",
             "multiwoz": f"{CONFIRM}/R24mwoz/tm_mem_c4_nrand"}
AGENT_SEEDS = (1, 2, 3, 4, 5)
AGENT_DEV = {"star": 0.6544585987261147, "abcd": 0.8417480998914224, "multiwoz": 0.35004756080989263}
FT_DEV = {"star": 0.7409, "abcd": 0.8727, "multiwoz": 0.4327}
FT_SETTINGS = {"model": "BAAI/bge-base-en-v1.5", "lr": 5e-05, "seed": 1, "epochs": 4, "batch": 32, "max_len": 256}
TFIDF_C = {"star": 4.0, "abcd": 16.0, "multiwoz": 1.0}
N_TEST = {"star": 6988, "abcd": 3608, "multiwoz": 7355}
DATASETS = ("star", "abcd", "multiwoz")


def agent_files(ds):
    return [os.path.join(AGENT_DIR[ds], f"seed{k}.npz") for k in AGENT_SEEDS]


def opt(argv, name):
    """The value after name in argv, or None."""
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def queue_starts(log_text):
    """[(name, args)] of every "start:" line, args as run_queue.sh logged them (the gate note cut off)."""
    out = []
    for line in log_text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[2].startswith("start: "):
            out.append((parts[1], parts[2][len("start: "):].rsplit(" (gate: ", 1)[0]))
    return out


def llm_settings_ok(res, c, ds, rows):
    argv = res.get("argv") or []
    mt = str(c["max_tokens"]) if c.get("max_tokens") else None
    return ("--test-registration" in argv and opt(argv, "--dataset") == ds and opt(argv, "--k") == str(c["k"])
            and ("--think" in argv) == bool(c["thinking"]) and ("--no-guidelines" in argv) == (not c["policy_text"])
            and opt(argv, "--max-tokens") == mt and opt(argv, "--seed") == "20261027" and opt(argv, "--n") == str(rows)
            and res.get("k") == c["k"] and bool(res.get("think")) == bool(c["thinking"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--choice", default=os.path.join(H, "runs/llm_star_choice.json"))
    ap.add_argument("--jobs", default=os.path.join(H, "jobs_reg.txt"))
    ap.add_argument("--queue-log", default=os.path.join(H, "queue.log"))
    ap.add_argument("--runs", default=os.path.join(H, "runs/reg"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-pins", action="store_true", help="tests only")
    a = ap.parse_args(argv)
    hard, flags = {}, {}
    fr = os.path.splitext(os.path.abspath(a.reg))[0] + ".frozen.json"
    hard["freeze record exists"] = os.path.exists(fr)
    if hard["freeze record exists"] and not a.skip_pins:
        hard["all pins unchanged"] = freeze_reg.main(["--check", fr]) == 0
    rows = {ds: v["rows"] for ds, v in json.load(open(a.rows))["datasets"].items()}
    choice = json.load(open(a.choice))["datasets"]
    pinned = {}
    for line in open(a.jobs).read().splitlines():
        if line.strip():
            name, args = line.split(" ", 1)
            pinned[name] = args
    starts = queue_starts(open(a.queue_log).read()) if os.path.exists(a.queue_log) else []
    for ds in DATASETS:
        files = agent_files(ds)
        hard[f"{ds}: agent seed files 1-5 exist"] = all(os.path.exists(f) for f in files)
        j1 = os.path.join(AGENT_DIR[ds], "seed1.json")
        dev1 = json.load(open(j1))["dev_acc"] if os.path.exists(j1) else None
        hard[f"{ds}: agent seed 1 dev accuracy equals the reference"] = dev1 is not None and abs(dev1 - AGENT_DEV[ds]) < 1e-12
        p = os.path.join(a.runs, f"llm_{ds}", "preds.jsonl")
        r = os.path.join(a.runs, f"llm_{ds}", "result.json")
        recs = [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
        want = sorted(random.Random(20261027).sample(range(N_TEST[ds]), min(rows[ds], N_TEST[ds])))
        hard[f"{ds}: LLM rows are the registered draw"] = sorted(x["i"] for x in recs) == want
        hard[f"{ds}: LLM rows hold no request error"] = bool(recs) and all(x.get("error") is None for x in recs)
        res = json.load(open(r)) if os.path.exists(r) else {}
        hard[f"{ds}: LLM result.json with n = rows"] = res.get("n") == rows[ds] == len(recs)
        hard[f"{ds}: LLM ran LLM*'s registered settings"] = llm_settings_ok(res, choice[ds]["llm_star"], ds, rows[ds])
        flags[f"{ds}: LLM invalid answers at most 0.5% of rows"] = (sum(x.get("pred") is None for x in recs)
                                                                   <= 0.005 * max(1, rows[ds]))
        for arm, fn in (("ft", "test_preds.npz"), ("tfidf", "test_preds.npz")):
            q = os.path.join(a.runs, f"{arm}_{ds}")
            rj = os.path.join(q, "result.json")
            rr = json.load(open(rj)) if os.path.exists(rj) else {}
            argv_arm = rr.get("argv") or []
            hard[f"{ds}: {arm} test predictions exist (not a dry run)"] = (os.path.exists(os.path.join(q, fn))
                                                                          and "--test-registration" in argv_arm
                                                                          and "--dry-run-dev" not in argv_arm)
            if arm == "ft":
                hard[f"{ds}: FT ran the registered settings"] = all(rr.get(k) == v for k, v in FT_SETTINGS.items())
                dev = rr.get("best_dev_acc")
                flags[f"{ds}: FT dev accuracy within 0.5 points of {FT_DEV[ds]}"] = (dev is not None
                                                                                    and abs(dev - FT_DEV[ds]) <= 0.005)
            else:
                hard[f"{ds}: TFIDF ran the registered C"] = (rr.get("C") == TFIDF_C[ds] and rr.get("dry_run_dev") is False
                                                             and rr.get("registration") is not None)
                flags[f"{ds}: TFIDF dev accuracy within 0.5 points of tfidf_lr.py's"] = bool(rr.get("dev_within_0.5_points"))
        names = (f"reg_ft_{ds}", f"reg_tfidf_{ds}", f"reg_llm_{ds}")
        hard[f"{ds}: the queue ran its job lines exactly as pinned"] = all(
            n in pinned and any(s[0] == n for s in starts) and all(s[1] == pinned[n] for s in starts if s[0] == n)
            for n in names)
    glob_ok = all(v for k, v in hard.items() if not k.startswith(tuple(f"{d}: " for d in DATASETS)))
    gate = {ds: glob_ok and all(v for k, v in hard.items() if k.startswith(f"{ds}: ")) for ds in DATASETS}
    out = {"hard": hard, "flags": flags, "gate": gate, "all_hard_pass": all(hard.values())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(out, h, indent=1)
    for k, v in hard.items():
        print(("ok    " if v else "FAIL  ") + k)
    for k, v in flags.items():
        print(("ok    " if v else "FLAG  ") + k)
    for ds in DATASETS:
        print(f"gate {ds}: " + ("open (reported)" if gate[ds] else "CLOSED: a hard check failed, no report for it"))
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
