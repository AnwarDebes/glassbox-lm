"""A2 power check on calib (finish pass 2026-10-25, agent tal; dev work, no rule, no run folder).

    cd studies/tally && nice -n 19 ../tm_language_model/.venv-tm/bin/python <this file>

Calib is cut from the standard training file by the same block procedure as dev-test (blocks of
100 sentences, ptb.py), so it is the closest split to dev-test that may be read. The dev-test
stream is dropped right after loading and never touched. For the arms of a2_run.py
(none, exchange, kmeans200, predefined, topic, kind_topic, orgroups, the 10 random codes), with the
codes that the development run 20261025T114718Z_a2_valid_dev saved in data/a2_codes/ (sha256 of
each code checked against its json) and the cache setting it chose on valid (half-life 100,
<eos> skipped, reset per block):
- the EM weights are fitted on one half of the calib blocks and the perplexity is read on the
  other half, both ways (2-fold by block), so no calib token is scored by weights fitted on it;
- reported: the gain of each arm over none, the margin of each candidate over the mean of the
  random codes, and the share of calib and valid positions whose bigram is unseen in train.
"""
import json
import os
import sys

import numpy as np

STUDY = os.getcwd()
sys.path.insert(0, os.path.join(STUDY, "src"))
from tally import ptb  # noqa: E402
from tally.floor import cache, codeeval, codes, mix  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

vocab, w2i, ids, blocks = ptb.load()
for d in (ids, blocks):
    del d["devtest"]
assert set(ids) == {"train", "calib", "valid"}
V, eos = len(vocab), w2i["<eos>"]
tr = ids["train"]

C = {}
for name in ["exchange", "kmeans200", "predefined", "topic", "orgroups"] + [f"random_s{s}" for s in range(1, 11)]:
    with open(os.path.join(codes.CODE_DIR, name + ".json")) as f:
        meta = json.load(f)
    c = np.load(os.path.join(codes.CODE_DIR, name + ".npy"))
    assert codes.sha256_bytes(np.ascontiguousarray(c, dtype=np.int32).tobytes()) == meta["code_sha256"], name
    assert meta["source"]["train_stream_sha256"] == codes.stream_hash(tr), name
    C[name] = c.astype(np.int64)

ex = [("exchange", l) for l in range(C["exchange"].shape[1])]
tp = [("topic", l) for l in range(C["topic"].shape[1])]
arms = {"none": [], "exchange": ex, "kmeans200": [("kmeans200", 0)],
        "predefined": [("predefined", l) for l in range(C["predefined"].shape[1])], "topic": tp,
        "kind_topic": ex + tp, "orgroups": [("orgroups", 0)]}
for s in range(1, 11):
    arms[f"random_s{s}"] = [(f"random_s{s}", l) for l in range(C[f"random_s{s}"].shape[1])]

kn5 = KN(tr, V, eos, order=5, mode="sent")
p_kn = {s: np.exp(kn5.logprob(ids[s])) for s in ("calib", "valid")}
cch = cache.Cache(V, 100.0, "block", (eos,))
p_c = {s: cch.prob(ids[s], blocks[s], p_kn[s])[0] for s in ("calib", "valid")}
needed = sorted({m for v in arms.values() for m in v})
P = {}
for code, l in needed:
    cm = ClassMember(tr, C[code][:, l], eos, 5, 0.5)
    P[(code, l)] = {s: np.exp(cm.logprob(ids[s])) for s in ("calib", "valid")}

cb = blocks["calib"]
ub = np.unique(cb)
half = np.isin(cb, ub[::2])                   # 2-fold by block: even and odd calib blocks
out, lp = {}, {}
for arm, mem in arms.items():
    cols = np.stack([p_kn["calib"], p_c["calib"]] + [P[m]["calib"] for m in mem], 1)
    l = np.empty(len(cb))
    for fit in (half, ~half):
        lam, _ = mix.em(cols[fit])
        l[~fit] = np.log(cols[~fit] @ lam)
    lp[arm] = l
    out[arm] = {"calib_crossfit_ppl": float(np.exp(-l.mean()))}
base = out["none"]["calib_crossfit_ppl"]
rnd = np.mean([out[f"random_s{s}"]["calib_crossfit_ppl"] for s in range(1, 11)])
for arm in arms:
    out[arm]["gain_vs_none"] = 1 - out[arm]["calib_crossfit_ppl"] / base
    out[arm]["margin_vs_random_mean"] = 1 - out[arm]["calib_crossfit_ppl"] / rnd
unseen = {s: float(codeeval.unseen_bigram_mask(tr, ids[s], eos, V).mean()) for s in ("calib", "valid")}
print(json.dumps({"calib_tokens": int(len(cb)), "calib_blocks": int(len(ub)), "random_mean_ppl": float(rnd),
                  "unseen_bigram_share": unseen, "arms": out}, indent=1))
