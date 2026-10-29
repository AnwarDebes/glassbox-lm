"""A2 power check from calib's own blocks (finish pass 2026-10-25, agent tal; dev work, no rule, no run folder).

    cd studies/tally && nice -n 19 ../tm_language_model/.venv-tm/bin/python -I <this file>

Answers review point m1 of DRAFT 2 (design lens): the standard error of the A2 statistic on a split
of 21 blocks cut from the training file, measured on calib (cut like dev-test) instead of scaled
from valid's 34 blocks. With the codes and the cache setting of the development runs (half-life
100, <eos> skipped, reset per block) and the EM weights of `none` and `exchange` cross-fitted over
two halves of the calib blocks (each half's weights read on the other half), it computes the
per-token bits difference bits(exchange) - bits(none) on calib and resamples calib's 21 blocks
(stats.block_bootstrap: seed 0 with 1,000 resamples, as a2_run.py, and seed 20261025 with 10,000).
Reported: the mean difference, the gain, the bootstrap SD in bits and in gain units, and from it
the gain needed for 80% power at the 5% line and the distance of the calib estimate from the line
in standard errors (with and without calib's own sampling error, sqrt 2). The dev-test stream is
dropped right after loading and never touched.
"""
import json
import os
import sys

import numpy as np

STUDY = os.getcwd()
sys.path.insert(0, os.path.join(STUDY, "src"))
from tally import ptb  # noqa: E402
from tally.floor import cache, codes, mix, stats  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

vocab, w2i, ids, blocks = ptb.load()
for d in (ids, blocks):
    del d["devtest"]
assert set(ids) == {"train", "calib", "valid"}
V, eos = len(vocab), w2i["<eos>"]
tr = ids["train"]
with open(os.path.join(codes.CODE_DIR, "exchange.json")) as f:
    meta = json.load(f)
C = np.load(os.path.join(codes.CODE_DIR, "exchange.npy"))
assert codes.sha256_bytes(np.ascontiguousarray(C, dtype=np.int32).tobytes()) == meta["code_sha256"]
assert meta["source"]["train_stream_sha256"] == codes.stream_hash(tr)
C = C.astype(np.int64)
kn5 = KN(tr, V, eos, order=5, mode="sent")
cb = blocks["calib"]
pk = np.exp(kn5.logprob(ids["calib"]))
pc = cache.Cache(V, 100.0, "block", (eos,)).prob(ids["calib"], cb, pk)[0]
mem = [np.exp(ClassMember(tr, C[:, l], eos, 5, 0.5).logprob(ids["calib"])) for l in range(C.shape[1])]
half = np.isin(cb, np.unique(cb)[::2])
lp = {}
for arm, ms in (("none", []), ("exchange", mem)):
    P = np.stack([pk, pc] + ms, 1)
    l = np.empty(len(cb))
    for fit in (half, ~half):
        lam, _ = mix.em(P[fit])
        l[~fit] = np.log(P[~fit] @ lam)
    lp[arm] = l
d = stats.bits_diff(lp["exchange"], lp["none"])
b0 = stats.block_bootstrap(d, cb, n_boot=1000, seed=0)
b1 = stats.block_bootstrap(d, cb, n_boot=10_000, seed=20261025)
mean = float(d.mean())
gain = 1 - 2 ** mean
line_bits = float(np.log2(0.95))
out = {"calib_tokens": int(len(cb)), "calib_blocks": int(len(np.unique(cb))), "mean_bits_diff": mean,
       "gain": gain, "boot_seed0_1000": b0, "boot_seed20261025_10000": b1, "line_bits": line_bits}
for key, b in (("seed0", b0), ("seed20261025", b1)):
    sd = b["boot_sd"]
    se_gain = sd * np.log(2) * 2 ** line_bits             # delta method at the line
    out[f"power_{key}"] = {
        "se_bits": sd, "se_gain_at_line": float(se_gain),
        "gain_needed_for_80pct_power": float(0.05 + 0.8416 * se_gain),
        "distance_calib_to_line_se": float((line_bits - mean) / sd),
        "distance_with_calib_error_se": float((line_bits - mean) / (np.sqrt(2) * sd))}
print(json.dumps(out, indent=1))
