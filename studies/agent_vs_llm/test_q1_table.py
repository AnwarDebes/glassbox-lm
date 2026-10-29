"""Checks of q1_table.py with a synthetic LLM file over the real ABCD dev rows (no model is run).

usage: studies/decision_layer/.venv-dl/bin/python -I test_q1_table.py
"""
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1_table as Q  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


with gzip.open(os.path.join(Q.F.DATA, "abcd", "examples_dev.json.gz"), "rt") as h:
    d = json.load(h)
B, gold = d["buttons"], np.array([r["y"] for r in d["rows"]], dtype=np.int16)
rng = np.random.default_rng(1)
idx = sorted(rng.choice(len(gold), 200, replace=False).tolist())
tmp = tempfile.mkdtemp()
try:
    with open(os.path.join(tmp, "llm.jsonl"), "w") as h:
        for j, i in enumerate(idx):  # LLM right on even j, wrong (invalid) on odd j; nearest demo always right
            h.write(json.dumps({"i": i, "gold": B[gold[i]], "pred": B[gold[i]] if j % 2 == 0 else None,
                                "demo_labels": [B[(gold[i] + 1) % 30]] * 9 + [B[gold[i]]]}) + "\n")
    top = np.stack([gold, (gold + 1) % 30], 1).astype(np.int16)  # a perfect "agent"
    np.savez(os.path.join(tmp, "tm.npz"), top_ids=top, top_scores=np.ones(top.shape, np.float32), gold=gold)
    out = os.path.join(tmp, "q1.json")
    Q.main(["--dataset", "abcd", "--llm", os.path.join(tmp, "llm.jsonl"), "--agent", os.path.join(tmp, "tm.npz"),
            "--out", out, "--draws", "500"])
    r = json.load(open(out))
    check("LLM accuracy 0.5 with 100 invalid answers", r["arms"]["llm"]["acc"] == 0.5 and r["llm_invalid"] == 100)
    check("nearest demo always right", r["arms"]["nearest_demo"]["acc"] == 1.0)
    check("demo majority always wrong", r["arms"]["demo_majority"]["acc"] == 0.0)
    check("perfect agent is +0.5 over the LLM, interval around it",
          abs(r["arms"]["agent_tm_c4"]["minus_llm"] - 0.5) < 1e-9 and r["arms"]["agent_tm_c4"]["ci95"][0] > 0.4)
finally:
    shutil.rmtree(tmp)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)
