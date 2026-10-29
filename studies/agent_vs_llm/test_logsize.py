"""Checks of the PLAN_logsize_dev.md scripts (no GPU, no LLM: a stub server answers).

usage: cd studies/agent_vs_llm && ../decision_layer/.venv-dl/bin/python -I \
           test_logsize.py
The agent wrapper is checked at its loader (the TM needs the GPU, which the registered run holds); its whole-split
equality with agent_preds.py is the first queued job (the plan's gate).
"""
import contextlib
import gzip
import http.server
import io
import json
import os
import re
import sys
import tempfile
import threading

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import logsize_subsets as S  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    fails += 0 if cond else 1


def write_train(d, ds, rows):
    os.makedirs(os.path.join(d, ds), exist_ok=True)
    with gzip.open(os.path.join(d, ds, "examples_train.json.gz"), "wt") as h:
        json.dump({"rows": rows, "buttons": ["a", "b"]}, h)


# ---- pools -----------------------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    rows = [{"convo_id": c, "y": c % 2} for c in range(1, 301) for _ in range(c % 3 + 1)]
    for ds in ("star", "abcd", "multiwoz"):
        write_train(d, ds, rows if ds != "abcd" else list(reversed(rows)))
    out = S.build(data=d, sizes=(4, 16, 64))
    o = out["star"]["order"]
    check("every dialogue once in the order", sorted(o) == list(range(1, 301)))
    check("the row order of the file does not change the pools", out["abcd"]["order"] == o)
    check("the same seed gives the same order", S.build(data=d, sizes=(4,))["star"]["order"] == o)
    k4, k16 = S.keep_ids(out, "star", 4), S.keep_ids(out, "star", 16)
    check("pools are nested", k4 < k16 and len(k4) == 4 and len(k16) == 16)
    check("rows per size counted from the rows", out["star"]["rows_per_size"]["16"] == sum(r["convo_id"] in k16 for r in rows))
    check("whole means every row", S.keep_ids(out, "star", "whole") is None and S.keep_ids(out, "star", "full") is None)
    try:
        S.keep_ids(out, "star", 301)
        check("a size above the dialogue count is refused", False)
    except ValueError:
        check("a size above the dialogue count is refused", True)
    p = os.path.join(d, "s.json")
    open(p, "w").write("{}")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            S.main(["--out", p])
        check("an existing subsets file is never overwritten", False)
    except FileExistsError:
        check("an existing subsets file is never overwritten", open(p).read() == "{}")

# ---- real pools of ABCD (two dialogues) for the wrappers ----------------------------------------------------------
import llm_next_action as L  # noqa: E402

tr_all, buttons = L.rows_of("abcd", "train")
two = sorted({r["convo_id"] for r in tr_all})[:2]
n_two = sum(r["convo_id"] in set(two) for r in tr_all)
tmp = tempfile.mkdtemp()
subsets = {"abcd": {"order": two + [c for c in sorted({r["convo_id"] for r in tr_all}) if c not in two][:62]}}
sp = os.path.join(tmp, "subsets.json")
json.dump(subsets, open(sp, "w"))

# ---- agent wrapper (loader only) ---------------------------------------------------------------------------------
import logsize_agent as LA  # noqa: E402

AB = LA.install(set(two))
tr, b2 = AB.load_examples("train", "abcd")
check("agent loader: train keeps only the pool's rows", len(tr) == n_two and {r["convo_id"] for r in tr} == set(two))
dv, _ = AB.load_examples("dev", "abcd")
dv0, _ = AB.load_examples.orig("dev", "abcd")
check("agent loader: dev rows untouched", len(dv) == len(dv0) and dv[0] == dv0[0])
check("agent loader: the action list untouched", b2 == buttons)
AB.load_examples = AB.load_examples.orig
LA.install(None)
tr_w, _ = AB.load_examples("train", "abcd")
check("agent loader: whole keeps every training row", len(tr_w) == len(tr_all))
AB.load_examples = AB.load_examples.orig
for what, bad in (("arguments without '--'", ["--size", "64", "--subsets", sp]),
                  ("the test split", ["--size", "64", "--", "--dataset", "abcd", "--out", tmp, "--test-registration", "R.md"]),
                  ("a dry run on test's path", ["--size", "64", "--", "--dataset", "abcd", "--out", tmp, "--dry-run-dev"])):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            LA.main(bad)
        check(f"agent wrapper refuses {what}", False)
    except SystemExit:
        check(f"agent wrapper refuses {what}", True)

# ---- LLM wrapper, end to end with a stub server ------------------------------------------------------------------
import logsize_llm as LL  # noqa: E402

seen = []


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(body)
        data = json.dumps({"choices": [{"message": {"content": "pull-up-account"}}],
                           "usage": {"prompt_tokens": 100}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"
out = os.path.join(tmp, "llm_N2", "abcd")
q1 = os.path.join(H, "runs/llm_q1_dev/abcd/preds.jsonl")
with contextlib.redirect_stdout(io.StringIO()):
    LL.main(["--size", "2", "--subsets", sp, "--", "--dataset", "abcd", "--n", "3", "--k", "50", "--seed", "2",
             "--exclude", q1, "--url", url, "--out", out])
recs = [json.loads(x) for x in open(os.path.join(out, "preds.jsonl"))]
n_ex = [len(re.findall(r"^Example \d+:", b["messages"][1]["content"], flags=re.M)) for b in seen]
check("LLM wrapper: 3 rows answered", len(recs) == 3 and len(seen) == 3)
check(f"LLM wrapper: k 50 but every prompt holds only the pool's {n_two} rows", n_ex == [n_two] * 3 and n_two < 50)
pool = json.load(open(os.path.join(out, "pool.json")))
check("LLM wrapper: pool.json names 2 dialogues", pool["n_dialogues"] == 2 and pool["n_train_rows"] == n_two)
ref = os.path.join(tmp, "ref")
L.rows_of = L.rows_of.orig
with contextlib.redirect_stdout(io.StringIO()):
    L.main(["--dataset", "abcd", "--n", "3", "--k", "50", "--seed", "2", "--exclude", q1, "--url", url, "--out", ref])
ref_recs = [json.loads(x) for x in open(os.path.join(ref, "preds.jsonl"))]
check("LLM wrapper: the same dev rows as the unwrapped client (same --seed and --exclude)",
      [r["i"] for r in recs] == [r["i"] for r in ref_recs])
check("LLM wrapper: the unwrapped client sends 50 examples", all(
    len(re.findall(r"^Example \d+:", b["messages"][1]["content"], flags=re.M)) == 50 for b in seen[3:]))
check("LLM wrapper: system prompt (actions, guidelines) unchanged",
      all(b["messages"][0]["content"] == seen[3]["messages"][0]["content"] for b in seen))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        LL.main(["--size", "2", "--subsets", sp, "--", "--dataset", "abcd", "--out", out, "--test-registration", "R.md"])
    check("LLM wrapper refuses the test split", False)
except SystemExit:
    check("LLM wrapper refuses the test split", True)
L.rows_of = getattr(L.rows_of, "orig", L.rows_of)
srv.shutdown()

# ---- TFIDF on a 64-dialogue pool of ABCD -------------------------------------------------------------------------
import logsize_tfidf as LT  # noqa: E402

rows_json = os.path.join(tmp, "rows.json")
dev_ids = list(range(0, 400, 20))
json.dump(dev_ids, open(rows_json, "w"))
od = os.path.join(tmp, "tfidf")
with contextlib.redirect_stdout(io.StringIO()):
    LT.main(["--dataset", "abcd", "--sizes", "64", "--subsets", sp, "--out-dir", od, "--rows", rows_json])
t = json.load(open(os.path.join(od, "abcd_64.json")))
keep64 = set(subsets["abcd"]["order"][:64])
pool_rows = [r for r in tr_all if r["convo_id"] in keep64]
gold = [int(dv0[i]["y"]) for i in dev_ids]
labs = {int(r["y"]) for r in pool_rows}
check("TFIDF: the pool's dialogues and rows", t["n_dialogues"] == 64 and t["n_train_rows"] == len(pool_rows))
check("TFIDF: labels in the pool", t["labels_in_pool"] == sorted(labs))
check("TFIDF: unseen-gold share recomputed", abs(t["unseen_gold_500"] - sum(g not in labs for g in gold) / len(gold)) < 1e-12)
check("TFIDF: accuracy on the rows recomputed from top1", abs(t["acc_500"] - sum(t["top1_dev"][i] == g for i, g in zip(dev_ids, gold)) / len(gold)) < 1e-12)
check("TFIDF: predictions only among the pool's labels", set(t["top1_dev"]) <= labs)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        LT.main(["--dataset", "abcd", "--sizes", "64", "--subsets", sp, "--out-dir", od, "--rows", rows_json])
    check("TFIDF: an existing result is never overwritten", False)
except FileExistsError:
    check("TFIDF: an existing result is never overwritten", True)

# ---- the table on synthetic files --------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import logsize_table as LTab  # noqa: E402

root = os.path.join(tmp, "root")
real_dev = {}
for ds in ("star", "abcd", "multiwoz"):
    dvj = json.load(gzip.open(os.path.join(LTab.DATA, ds, "examples_dev.json.gz"), "rt"))
    real_dev[ds] = (dvj["buttons"], np.array([int(r["y"]) for r in dvj["rows"]]))


def wjson(p, obj, lines=False):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as h:
        if lines:
            h.writelines(json.dumps(x) + "\n" for x in obj)
        else:
            json.dump(obj, h)


acc_plan = {"64": (0.40, 0.60), "256": (0.50, 0.60), "1024": (0.62, 0.60), "4096": (0.64, 0.60), "whole": (0.66, 0.64)}
for ds in ("star", "abcd", "multiwoz"):
    btn, g = real_dev[ds]
    idx = np.arange(0, 5000, 10)[:500] % len(g)
    idx = np.array(sorted(set(idx.tolist())))[:500]
    wrong = lambda y: (y + 1) % len(btn)  # noqa: E731
    base = [{"i": int(i), "gold": btn[g[i]], "pred": btn[g[i]], "error": None, "s": 1.0} for i in idx]
    wjson(os.path.join(root, f"runs/llm_q3_L35k50/{ds}/preds.jsonl"), base, lines=True)
    for size, (aa, la) in acc_plan.items():
        na, nl = int(aa * len(idx)), int(la * len(idx))
        top = np.repeat(g[:, None], 5, 1).copy()
        top[:, 0] = g
        top[idx[na:], 0] = wrong(g[idx[na:]])
        pa, pl, pt = LTab.paths(root, ds, size)
        os.makedirs(os.path.dirname(pa), exist_ok=True)
        np.savez(pa, top_ids=top, gold=g)
        recs = [{"i": int(i), "gold": btn[g[i]], "pred": btn[g[i]] if k < nl else (None if k == nl else btn[wrong(g[i])]),
                 "error": None, "s": 2.0} for k, i in enumerate(idx)]
        if size != "whole":
            wjson(pl, recs, lines=True)
        else:
            wjson(pl, [dict(r, s=1.0) for r in recs], lines=True)
        t1 = g.copy()
        wjson(pt, {"n_dialogues": size, "n_train_rows": 1, "unseen_gold_500": 0.1, "top1_dev": t1.tolist(),
                   "equal_to_dry_run": True})
wjson(os.path.join(root, "runs/logsize/agent_abcd_whole/result.json"), {"dev_check": {"equal": True, "got": 1, "expected": 1}})
wjson(os.path.join(root, "runs/logsize/agent_multiwoz_whole/result.json"), {"arms": {"tm": {"dev_acc": 0.35}}})
with contextlib.redirect_stdout(io.StringIO()):
    out = LTab.table(root, 2000)
st = {r["size"]: r for r in out["datasets"]["star"]["rows"]}
check("table: accuracies as built", abs(st["64"]["acc_AGENT"] - 0.40) < 1e-9 and abs(st["64"]["acc_LLM*"] - 0.60) < 1e-9)
check("table: an invalid LLM answer counts as wrong and is counted", st["64"]["llm_invalid"] == 1)
check("table: LLM* ahead at 64 (interval below 0)", st["64"]["label"] == "LLM* ahead")
check("table: N_reach is 1024 (AGENT - LLM* >= 0 from 1024 on)", out["datasets"]["star"]["N_reach"] == "1024")
check("table: TFIDF read (equal to the dry run)", abs(st["whole"]["acc_TFIDF"] - 1.0) < 1e-9)
check("table: MultiWOZ rerun accuracy written beside e84's", out["datasets"]["multiwoz"]["rerun_whole_dev_acc_all_rows"] == 0.35)
check("table: n_reach helper", LTab.n_reach([("64", -1), ("256", 0.0), ("1024", None), ("whole", 0.01)]) == "whole"
      and LTab.n_reach([("64", 1), ("whole", -0.01)]) == "not reached")
wjson(os.path.join(root, "runs/logsize/agent_abcd_whole/result.json"), {"dev_check": {"equal": False, "got": 0.8, "expected": 0.84}})
wjson(os.path.join(root, "runs/logsize/tfidf/abcd_whole.json"), {"n_dialogues": "whole", "n_train_rows": 1, "unseen_gold_500": 0,
                                                              "top1_dev": [0], "equal_to_dry_run": False})
recs = [json.loads(x) for x in open(os.path.join(root, "runs/logsize/llm_N256/abcd/preds.jsonl"))][:-1]
wjson(os.path.join(root, "runs/logsize/llm_N256/abcd/preds.jsonl"), recs, lines=True)
with contextlib.redirect_stdout(io.StringIO()):
    out2 = LTab.table(root, 500)
ab = {r["size"]: r for r in out2["datasets"]["abcd"]["rows"]}
mw = {r["size"]: r for r in out2["datasets"]["multiwoz"]["rows"]}
check("table: a failed gate blocks every AGENT(N) and MultiWOZ's rerun", "acc_AGENT" not in ab["64"]
      and "agent_note" in ab["64"] and "acc_AGENT" not in mw["whole"])
check("table: a failed gate keeps STAR's and ABCD's earlier whole-split agent", "acc_AGENT" in ab["whole"])
check("table: TFIDF not read when it differs from the dry run", "acc_TFIDF" not in ab["whole"] and "tfidf_note" in ab["whole"])
check("table: an incomplete LLM run is not read", "acc_LLM*" not in ab["256"] and "llm_note" in ab["256"])
md = LTab.markdown(out2)
check("table: markdown names the failed gate", "NOT passed" in md)
op = os.path.join(tmp, "tab")
with contextlib.redirect_stdout(io.StringIO()):
    LTab.main(["--root", root, "--out", op, "--draws", "200"])
try:
    with contextlib.redirect_stdout(io.StringIO()):
        LTab.main(["--root", root, "--out", op, "--draws", "200"])
    check("table: never overwrites", False)
except FileExistsError:
    check("table: never overwrites", True)

# ---- the gate script -------------------------------------------------------------------------------------------
import logsize_gate as G  # noqa: E402

gp = os.path.join(tmp, "gate", "result.json")
with contextlib.redirect_stdout(io.StringIO()):
    r_missing = G.main(["--result", gp])
    wjson(gp, {"dev_check": {"equal": True, "got": 0.84, "expected": 0.84}})
    r_ok = G.main(["--result", gp])
    wjson(gp, {"dev_check": {"equal": False, "got": 0.83, "expected": 0.84}})
    r_bad = G.main(["--result", gp])
    wjson(gp, {"arms": {}})
    r_nocheck = G.main(["--result", gp])
check("gate script: 1 when not run, 0 when equal, 1 when not equal or unchecked",
      (r_missing, r_ok, r_bad, r_nocheck) == (1, 0, 1, 1))

print(f"{fails} failed")
print("PASSED" if fails == 0 else "FAILED")
sys.exit(1 if fails else 0)
