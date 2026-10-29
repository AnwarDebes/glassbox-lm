"""Checks of cascade.py on synthetic predictions over the real ABCD dev rows (no model is run).

usage: studies/decision_layer/.venv-dl/bin/python -I test_cascade.py
"""
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade as C  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


with gzip.open(os.path.join(C.DATA, "abcd", "examples_dev.json.gz"), "rt") as h:
    rows = json.load(h)["rows"]
gold = np.array([r["y"] for r in rows], dtype=np.int16)
n = len(gold)
rng = np.random.default_rng(0)
margin = rng.random(n)
# agent: right on every row with margin > 0.3, wrong otherwise; partner: right on 80% of rows at random
agent_top = np.where(margin > 0.3, gold, (gold + 1) % 30)
partner_ok = rng.random(n) < 0.8
partner_top = np.where(partner_ok, gold, (gold + 2) % 30)
tmp = tempfile.mkdtemp()
try:
    ta = np.stack([agent_top, (agent_top + 3) % 30], 1).astype(np.int16)
    ts = np.stack([margin + 1.0, np.ones(n)], 1).astype(np.float32)  # top1 - top2 = margin
    np.savez(os.path.join(tmp, "tm.npz"), top_ids=ta, top_scores=ts, gold=gold)
    tp = np.stack([partner_top, (partner_top + 4) % 30], 1).astype(np.int16)
    np.savez(os.path.join(tmp, "p.npz"), top_ids=tp, top_probs=np.ones((n, 2), np.float16), gold=gold)
    out = os.path.join(tmp, "o.json")
    C.main(["--dataset", "abcd", "--agent", os.path.join(tmp, "tm.npz"), "--partner", os.path.join(tmp, "p.npz"),
            "--out", out, "--draws", "200"])
    r = json.load(open(out))
    check("agent accuracy is the share of margins above 0.3", abs(r["agent_acc"] - (margin > 0.3).mean()) < 1e-9)
    # largest coverage within 0.5 points of the partner on the fit half: 0.7 + 0.8 (1 - c) >= 0.795, so c near 88%
    check("the fit half picks the largest coverage at no loss (80-95%)",
          all(80 <= x["coverage_fit"] <= 95 for x in r["readings"]))
    check("the score half stays within a few points of the partner", all(x["diff"] > -0.06 for x in r["readings"]))
    check("claim_holds follows the rule of the note", r["claim_holds"] == all(
        x["diff"] >= -0.005 and x["ci95"][0] > -0.015 for x in r["readings"]))
    a_conv = C.halves(np.array([x["convo_id"] for x in rows]))
    convo = np.array([x["convo_id"] for x in rows])
    check("halves keep whole dialogues", not (set(convo[a_conv]) & set(convo[~a_conv])))
    # a partner that is always right: the cascade must not claim no loss at high coverage
    np.savez(os.path.join(tmp, "p2.npz"), top_ids=np.stack([gold, gold], 1), top_probs=np.ones((n, 2), np.float16),
             gold=gold)
    C.main(["--dataset", "abcd", "--agent", os.path.join(tmp, "tm.npz"), "--partner", os.path.join(tmp, "p2.npz"),
            "--out", out, "--draws", "200"])
    r2 = json.load(open(out))
    check("a perfect partner leaves the agent only rows it gets right",
          all(x["cascade_acc"] >= 0.99 for x in r2["readings"]))
    # the LLM path: only rows with an answer
    sel = sorted(rng.choice(n, 100, replace=False).tolist())
    with open(os.path.join(tmp, "llm.jsonl"), "w") as h:
        bl = json.load(gzip.open(os.path.join(C.DATA, "abcd", "examples_dev.json.gz"), "rt"))["buttons"]
        for i in sel:
            h.write(json.dumps({"i": i, "gold": bl[gold[i]], "pred": bl[gold[i]]}) + "\n")
    C.main(["--dataset", "abcd", "--agent", os.path.join(tmp, "tm.npz"), "--partner-llm",
            os.path.join(tmp, "llm.jsonl"), "--out", out, "--draws", "200"])
    r3 = json.load(open(out))
    check("the LLM path keeps only answered rows", r3["n"] == 100 and r3["partner_acc"] == 1.0)
finally:
    shutil.rmtree(tmp)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)
