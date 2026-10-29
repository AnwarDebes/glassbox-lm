#!/usr/bin/env python
"""Tests for b3_c4_feats.py on a synthetic word table and synthetic sentences (a plain script, no pytest; CPU only):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_b3_c4_feats.py

Checks the table reading (a zero row has no word), the token rows, the concepts and thresholds, the bits against a
brute-force recomputation, the CLI fit / encode with --start and --limit, determinism and the refusal to overwrite.
Each run has its own folder in tu/tests/runs/<UTC stamp>_b3c4feats/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
import b3_c4_feats as F

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3c4feats")
results = {"pass": 0, "fail": 0}
torch.set_num_threads(2)


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


RUNS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5)
words = [f"w{i}" for i in range(60)] + ["dead", "card", "n't", "?"]
codes = rng.standard_normal((len(words), 16)).astype(np.float16)
codes[words.index("dead")] = 0
tp = RUNS / "table.npz"
np.savez(tp, codes=codes, words=np.asarray(words), counts=np.ones(len(words), np.int64), vocab_ids=np.arange(len(words), dtype=np.int32))
M, ws, row = F.read_table(tp)
check("a zero row has no word", "dead" not in row and "card" in row and len(row) == len(words) - 1)
ids, info = F.token_rows(["W1 w2 card", "dead zzz", "don't stop?"], row)
check("tokens map to table rows (lowercased, unknown and dead words dropped)",
      ids[0] == [row["w1"], row["w2"], row["card"]] and ids[1] == [], str(ids))
check("contractions split as wordtab_feats does", ids[2] == [row["n't"], row["?"]], str(ids[2]))
check("the OOV share and the texts without a token", info["texts_without_token"] == 1 and 0 < info["oov_token_share"] < 1, str(info))

sents = [" ".join(f"w{j}" for j in rng.integers(0, 60, rng.integers(1, 9))) for _ in range(400)] + ["zzz qqq", "card"]
sp = RUNS / "sentences.txt"
sp.write_text("\n".join(sents) + "\n")
ids, _ = F.token_rows(sents, row)
c = F.fit_concepts(M, ids, 8, "cpu", 300)
check("K concepts, unit length", c["Cn"].shape == (8, 16) and np.allclose(np.linalg.norm(c["Cn"], axis=1), 1, atol=1e-5))
check("the members cover the vocabulary", sum(c["members"]) == c["vocab_rows"] == len({r for x in ids for r in x}))
check("thresholds are sorted and unique per concept", all(t == sorted(set(t)) for t in c["thresholds"]), str(c["thresholds"][:2]))
c2 = F.fit_concepts(M, ids, 8, "cpu", 300)
check("fit is deterministic (generator seeded 0)", np.array_equal(c["Cn"], c2["Cn"]) and c["thresholds"] == c2["thresholds"])
Xp = F.encode_bits(M, c["Cn"], c["thresholds"], ids, "cpu", bs=64)
n_bits = sum(len(t) for t in c["thresholds"])
bits = np.unpackbits(Xp, axis=1)[:, :n_bits]
En = M.astype(np.float64) / np.linalg.norm(M.astype(np.float64), axis=1, keepdims=True).clip(1e-12)
ref, near = [], []
for x in ids:
    mx = (En[x] @ c["Cn"].astype(np.float64).T).max(0) if x else np.zeros(8)
    ref.append(np.concatenate([(mx[k] >= np.asarray(t)) for k, t in enumerate(c["thresholds"])]))
    near.append(np.concatenate([np.abs(mx[k] - np.asarray(t)) < 1e-5 for k, t in enumerate(c["thresholds"])]))
ref, near = np.asarray(ref, np.uint8), np.asarray(near)
# the 0 and 100 percentiles are members' own cosines, so a text holding that word sits on the threshold; float rounding
# may decide those bits either way, so they are excluded from the comparison and counted
off = (bits != ref) & ~near
check(f"bits equal a float64 brute-force recomputation away from the thresholds ({int(near.sum())} bits within 1e-5 left out)",
      bits.shape == ref.shape and not off.any(), f"{bits.shape} vs {ref.shape}, {int(off.sum())} differ")
check("a text with no token has its max cosine 0 (bits only for thresholds <= 0)", np.array_equal(bits[400], ref[400]))
mx64 = np.stack([(En[x] @ c["Cn"].astype(np.float64).T).max(0) if x else np.zeros(8) for x in ids[:300]])
qs = np.linspace(0, 100, 9)[1:-1]
th_ref = [np.unique(np.percentile(mx64[:, k], qs)) for k in range(8)]
check("the thresholds are the existsT thermometer: Thermo(8) quantiles of the first 300 texts' max cosines",
      all(len(a) == len(b) and np.allclose(a, b, atol=1e-4) for a, b in zip(c["thresholds"], th_ref)), str(c["thresholds"][:1]))
check("the bits are dense as existsT is (on-share 0.35-0.65 on the fit rows)", 0.35 < bits[:300].mean() < 0.65, str(bits[:300].mean()))

PY = [sys.executable, "-I", "-B", str(TU / "b3_c4_feats.py")]
cp = RUNS / "concepts.npz"
p = subprocess.run(PY + ["fit", "--sentences", str(sp), "--table", str(tp), "--out", str(cp), "--K", "8", "--thermo-rows", "300"],
                   capture_output=True, text=True, timeout=300, check=False)
check("CLI fit: exit 0", p.returncode == 0, p.stderr[-600:])
z = np.load(cp) if cp.exists() else None
check("CLI fit: the same concepts as in-process", z is not None and np.array_equal(z["Cn"], c["Cn"])
      and json.loads(str(z["thresholds"])) == c["thresholds"])
meta = json.loads(str(z["meta"])) if z is not None else {}
check("CLI fit: meta has the hashes, lines and bit count", meta.get("lines") == len(sents) and meta.get("n_bits") == n_bits
      and len(meta.get("sentences_sha16", "")) == 16 and len(meta.get("table_sha16", "")) == 16, str(meta)[:300])
xp = RUNS / "x.npz"
p = subprocess.run(PY + ["encode", "--concepts", str(cp), "--table", str(tp), "--sentences", str(sp), "--start", "10",
                         "--limit", "50", "--out", str(xp)], capture_output=True, text=True, timeout=300, check=False)
check("CLI encode: exit 0", p.returncode == 0, p.stderr[-600:])
if p.returncode == 0:
    zx = np.load(xp)
    check("CLI encode: rows 10-59 equal the in-process bits", int(zx["n_features"]) == n_bits and np.array_equal(zx["X"], Xp[10:60]))
p2 = subprocess.run(PY + ["encode", "--concepts", str(cp), "--table", str(tp), "--sentences", str(sp), "--out", str(xp)],
                    capture_output=True, text=True, timeout=300, check=False)
check("CLI encode: an existing output is refused (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr)
p3 = subprocess.run(PY + ["fit", "--sentences", str(sp), "--table", str(tp), "--out", str(cp)], capture_output=True, text=True,
                    timeout=300, check=False)
check("CLI fit: an existing output is refused (exit 2)", p3.returncode == 2, p3.stderr)

print(f"b3_c4_feats tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
