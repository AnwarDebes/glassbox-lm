"""Checks of hybrid_table.py on synthetic answers over the real ABCD dev rows (no model is run).

usage: studies/decision_layer/.venv-dl/bin/python -I test_hybrid_table.py
"""
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_table as T  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


with gzip.open(os.path.join(T.F.DATA, "abcd", "examples_dev.json.gz"), "rt") as h:
    d = json.load(h)
B, gold = d["buttons"], np.array([r["y"] for r in d["rows"]], dtype=np.int16)
rng = np.random.default_rng(3)
idx = sorted(rng.choice(len(gold), 400, replace=False).tolist())
tmp = tempfile.mkdtemp()
try:
    # hybrid k3 right on 90% of rows, k5 on 80%, LLM alone on 60%; agent right on 70% (rows by position)
    for arm, share in (("k3", 0.9), ("k5", 0.8), ("alone", 0.6)):
        os.makedirs(os.path.join(tmp, f"abcd_{arm}"))
        with open(os.path.join(tmp, f"abcd_{arm}", "preds.jsonl"), "w") as h:
            for j, i in enumerate(reversed(idx)):  # written out of order on purpose
                right = (j % 10) < share * 10
                h.write(json.dumps({"i": i, "pred": B[gold[i]] if right else B[(gold[i] + 1) % 30],
                                    "fallback": arm != "alone" and j % 50 == 0}) + "\n")
    top = np.stack([np.where(np.arange(len(gold)) % 10 < 7, gold, (gold + 2) % 30), (gold + 5) % 30, gold], 1)
    np.savez(os.path.join(tmp, "tm.npz"), top_ids=top.astype(np.int16), top_scores=np.ones(top.shape, np.float32),
             gold=gold)
    out = os.path.join(tmp, "o.json")
    T.main(["--dataset", "abcd", "--dir", tmp, "--agent", os.path.join(tmp, "tm.npz"), "--out", out, "--draws", "500"])
    r = json.load(open(out))
    check("accuracies are the planted shares", abs(r["acc"]["hybrid_k3"] - 0.9) < 0.01 and
          abs(r["acc"]["llm_alone"] - 0.6) < 0.01)
    check("gold always in the agent's top 3", r["gold_in_agent_top3"] == 1.0)
    check("k3 minus LLM is about +0.30 with a positive interval",
          abs(r["k3_minus_llm"]["diff"] - 0.3) < 0.01 and r["k3_minus_llm"]["ci95"][0] > 0)
    check("fallbacks are counted", r["fallbacks"]["k3"] == 8 and r["fallbacks"]["alone"] == 0)
    check("the rule needs both differences", r["rule_met_this_dataset"] ==
          (r["k3_minus_agent"]["diff"] >= 0.02 and r["k3_minus_agent"]["ci95"][0] > 0))
finally:
    shutil.rmtree(tmp)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)
