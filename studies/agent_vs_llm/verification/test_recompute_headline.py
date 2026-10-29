"""Checks of recompute_headline.py on synthetic files (no real test data is read).

usage: studies/decision_layer/.venv-dl/bin/python -I test_recompute_headline.py
"""
import contextlib
import copy
import gzip
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_headline as R  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp())
data, confirm, head = tmp / "data", tmp / "confirm", tmp / "head"
ds, N, n = "abcd", 300, 200
buttons = ["b0", "b1", "b2", "b3", "b4"]
rng = np.random.default_rng(5)
convo = np.repeat(np.arange(1000, 1060), 5)          # 60 dialogues x 5 turns
k = np.tile(np.arange(5), 60)
y = rng.integers(0, 5, N)
(data / ds).mkdir(parents=True)
with gzip.open(data / ds / "examples_test.json.gz", "wt") as h:
    json.dump({"rows": [{"y": int(y[i]), "convo_id": int(convo[i]), "k": int(k[i])} for i in range(N)],
               "buttons": buttons}, h)
rows = sorted(random.Random(20261027).sample(list(range(N)), n))
wrong = (y + 1) % 5


def pred_with(acc, seed):
    r = np.random.default_rng(seed).random(N)
    return np.where(r < acc, y, wrong)


sd = confirm / R.SEED_DIRS[ds]
sd.mkdir(parents=True)
agent = {s: pred_with(0.80, 10 + s) for s in range(1, 6)}
for s, p in agent.items():
    np.savez(sd / f"seed{s}.npz", pred=p, y=y, convo_id=convo, k=k)
llm = pred_with(0.60, 99)
reg = head / "runs/reg"
(reg / f"llm_{ds}").mkdir(parents=True)
recs = [{"i": int(i), "gold": buttons[y[i]], "pred": (None if j == 0 else buttons[llm[i]]), "error": None}
        for j, i in enumerate(rows)]
(reg / f"llm_{ds}" / "preds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
ft = pred_with(0.95, 7)
tf = pred_with(0.78, 8)
for arm, p in (("ft", ft), ("tfidf", tf)):
    (reg / f"{arm}_{ds}").mkdir(parents=True)
    np.savez(reg / f"{arm}_{ds}" / "test_preds.npz", top_ids=np.stack([p] * 5, 1), gold=y)

# the report's point estimates, computed here without the tool
idx = np.array(rows)
a1 = (agent[1][idx] == y[idx]).astype(float)
lo = np.array([0.0 if j == 0 else float(llm[i] == y[i]) for j, i in enumerate(rows)])
f_ok = (ft[idx] == y[idx]).astype(float)
t_ok = (tf[idx] == y[idx]).astype(float)
A = np.stack([(agent[s][idx] == y[idx]).astype(float) for s in range(1, 6)])
report = {"acc_agent": a1.mean(), "acc_llm": lo.mean(), "acc_ft": f_ok.mean(),
          "agent_minus_llm": (a1 - lo).mean(), "ci_agent_minus_llm": [0.12, 0.28], "ci90_agent_minus_llm": [0.13, 0.27],
          "ft_minus_agent": (f_ok - a1).mean(), "ci_ft_minus_agent": [0.08, 0.22],
          "NI_met": True, "NI2_met": True, "SUP_met": True, "FT_met": True,
          "tfidf_no_rule": {"acc_tfidf": t_ok.mean(), "tfidf_minus_llm": (t_ok - lo).mean(),
                            "agent_minus_tfidf": (a1 - t_ok).mean()},
          "two_level": {"agent_minus_llm_mean": (A - lo).mean(), "ci": [0.1, 0.3]}}
report = json.loads(json.dumps(report, default=float))


def run(rep, extra=()):
    (reg / f"report_{ds}.json").write_text(json.dumps(rep))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["--headline", str(head), "--data", str(data), "--confirm", str(confirm), "--draws", "400",
                     "--datasets", ds, "--n-rows", json.dumps({ds: n}), *extra])
    return rc, json.loads(buf.getvalue())[ds]


rc, out = run(report)
mine = out["recompute"]
check("MATCH on a consistent report (exit 0)", rc == 0 and out["verdict"] == "MATCH")
check("rows: the registered draw regenerated", mine["n_rows"] == n and not mine["problems"])
check("an invalid LLM answer counts as wrong and is counted", mine["llm_invalid"] == 1 and abs(mine["acc_llm"] - lo.mean()) < 1e-12)
check("the 5-seed mean difference", abs(mine["two_level"]["agent_minus_llm_mean"] - (A - lo).mean()) < 1e-12)
check("rules from the dialogue bootstrap (clear-cut here)", mine["NI_met"] and mine["NI2_met"] and mine["SUP_met"] and mine["FT_met"])
check("the 90% interval lies inside the 95% one", mine["ci_agent_minus_llm"][0] <= mine["ci90_agent_minus_llm"][0]
      and mine["ci90_agent_minus_llm"][1] <= mine["ci_agent_minus_llm"][1])
bad = copy.deepcopy(report)
bad["agent_minus_llm"] += 0.001
rc, out = run(bad)
check("a point estimate off by 0.001 -> MISMATCH (exit 1)", rc == 1 and out["verdict"] == "MISMATCH")
bad = copy.deepcopy(report)
bad["NI_met"] = False
rc, out = run(bad)
check("a different rule outcome far from its bound -> MISMATCH", out["verdict"] == "MISMATCH"
      and out["compare"]["rules"]["NI_met"]["status"] == "DIFFERENT")
near = copy.deepcopy(report)
near["SUP_met"] = False
near["ci_agent_minus_llm"] = [0.001, 0.28]
rc, out = run(near)
check("a different rule outcome with the report's lower end within 0.003 of the bound -> NEAR-BOUND, not MISMATCH",
      out["compare"]["rules"]["SUP_met"]["status"] == "NEAR-BOUND" and out["verdict"].startswith("MATCH (NEAR-BOUND"))
(reg / f"llm_{ds}" / "preds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs[:-1]))
rc, out = run(report)
check("an LLM run missing a registered row -> MISMATCH with the problem named", out["verdict"] == "MISMATCH"
      and any("not the registered draw" in p for p in out["recompute"]["problems"]))
(reg / f"llm_{ds}" / "preds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
np.savez(sd / "seed3.npz", pred=agent[3], y=y, convo_id=convo, k=np.roll(k, 1))
rc, out = run(report)
check("an agent seed file out of line with the test file -> MISMATCH", out["verdict"] == "MISMATCH"
      and any("seed 3: k differs" in p for p in out["recompute"]["problems"]))
np.savez(sd / "seed3.npz", pred=agent[3], y=y, convo_id=convo, k=k)
(reg / f"report_{ds}.json").unlink()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    R.main(["--headline", str(head), "--data", str(data), "--confirm", str(confirm), "--datasets", ds])
check("no report -> not recomputed", json.loads(buf.getvalue())[ds]["verdict"] == "no report")
# the dialogue bootstrap against a direct loop
d = np.array([1.0, 0, 1, 1, 0, 0, 1, 0])
c = np.array([1, 1, 2, 2, 3, 3, 4, 4])
b = R.cluster_boot(d, c, 2000, np.random.default_rng(1))
check("dialogue bootstrap: mean of draws near the row mean, spread > 0", abs(b.mean() - d.mean()) < 0.03 and b.std() > 0.05)
print(f"{fails} failed")
print("PASSED" if fails == 0 else "FAILED")
sys.exit(1 if fails else 0)
