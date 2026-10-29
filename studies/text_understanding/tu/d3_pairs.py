#!/usr/bin/env python
"""Step 3d, the pair counts (tu/PLAN_3d.md "Pair list"): ordered adjacent word pairs in 3a's training sentences, C4
data only (no intent set is read here).

    python -I -B d3_pairs.py --selftest   # synthetic test (no files)
    python -I -B d3_pairs.py              # tu/d3/pair_counts_c5.npz + tu/d3/pair_counts_summary.json; refuses to overwrite

A pair is two tokens next to each other in glove_feats.glove_tokens(sentence), both in the live word list of
ppw_S1000 (the same list as a3B's). Nothing is skipped: an unknown token between two words breaks the pair. Pairs are
counted over tu/a3/c4_sent_train_f.txt only (1,999,932 sentences); every pair seen at least 5 times is kept, sorted by
count (descending; ties by the pair key w1 * V + w2, ascending). The summary gives the count distribution that fixes
the cut (PLAN_3d.md); the cut itself is applied by d3_fit.py.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
A3 = HERE / "a3"
D3 = HERE / "d3"
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
MIN_KEEP = 5
CUTS = (5, 10, 20, 50, 100, 200, 500, 1000, 5000)
TOPS = (10_000, 25_000, 50_000, 100_000, 200_000, 400_000)


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pair_keys(toks: list[str], row: dict[str, int], V: int) -> list[int]:
    """Keys w1 * V + w2 of the adjacent pairs whose two tokens are both live words."""
    ids = [row.get(t, -1) for t in toks]
    return [a * V + b for a, b in itertools.pairwise(ids) if a >= 0 and b >= 0]


def count_pairs(texts, tokenize, row: dict[str, int], V: int, min_keep: int = MIN_KEEP):
    keys = []
    n_occ = 0
    for s in texts:
        k = pair_keys(tokenize(s), row, V)
        n_occ += len(k)
        keys.append(np.asarray(k, dtype=np.int64))
    allk = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    u, c = np.unique(allk, return_counts=True)
    order = np.lexsort((u, -c))  # count descending, then key ascending
    u, c = u[order], c[order]
    summary = {"sentences": len(texts), "pair_occurrences": int(n_occ), "distinct_pairs": len(u),
               "pairs_with_count_at_least": {str(m): int((c >= m).sum()) for m in CUTS},
               "occurrence_share_of_top": {str(t): round(float(c[:t].sum()) / max(1, n_occ), 4) for t in TOPS},
               "count_at_rank": {str(t): int(c[t - 1]) if t <= len(c) else 0 for t in TOPS}}
    keep = c >= min_keep
    return u[keep], c[keep], summary


def selftest() -> None:
    row = {"turn": 0, "off": 1, "the": 2, "lights": 3, "on": 4}
    V = 5
    tok = str.split
    assert pair_keys(tok("turn off the lights"), row, V) == [0 * 5 + 1, 1 * 5 + 2, 2 * 5 + 3]
    assert pair_keys(tok("turn xyz off"), row, V) == [], "an unknown token must break the pair"
    assert pair_keys(tok("turn"), row, V) == [] and pair_keys([], row, V) == []
    texts = ["turn off the lights", "turn on the lights", "turn off", "the lights", "zzz"]
    u, c, s = count_pairs(texts, tok, row, V, min_keep=1)
    got = {(int(k) // V, int(k) % V): int(n) for k, n in zip(u, c)}
    assert got == {(0, 1): 2, (1, 2): 1, (2, 3): 3, (0, 4): 1, (4, 2): 1}, got
    assert list(c) == sorted(c, reverse=True) and int(u[0]) == 2 * 5 + 3
    assert s["pair_occurrences"] == 8 and s["distinct_pairs"] == 5 and s["sentences"] == 5
    ties = [int(k) for k, n in zip(u, c) if n == 1]
    assert ties == sorted(ties), "ties must be ordered by key"
    u2, c2, _ = count_pairs(texts, tok, row, V, min_keep=2)
    assert len(u2) == 2 and set(c2.tolist()) == {2, 3}
    assert (1, 0) not in got and pair_keys(tok("off turn"), row, V) == [1 * 5 + 0], "the order of a pair must matter"
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    assert sys.argv[1:] == [], sys.argv
    sys.path.insert(0, str(EMB))
    import glove_feats as GF

    t0 = time.time()
    D3.mkdir(exist_ok=True)
    out, summ = D3 / "pair_counts_c5.npz", D3 / "pair_counts_summary.json"
    for p in (out, summ):
        assert not p.exists(), f"{p} exists"
    z = np.load(TABLE)
    W0 = z["codes"].astype(np.float32)
    dead = np.abs(W0).sum(1) == 0
    row = {str(w): i for i, w in enumerate(z["words"]) if not dead[i]}
    V = len(z["words"])
    src = A3 / "c4_sent_train_f.txt"
    texts = src.read_text(encoding="utf-8").split("\n")[:-1]
    u, c, s = count_pairs(texts, GF.glove_tokens, row, V)
    np.savez(out, w1=(u // V).astype(np.int32), w2=(u % V).astype(np.int32), counts=c.astype(np.int64), V=np.asarray(V))
    s.update({"min_keep": MIN_KEEP, "kept": len(u), "words": V, "dead_words": int(dead.sum()),
              "top_pairs": [[str(z["words"][int(k) // V]), str(z["words"][int(k) % V]), int(n)] for k, n in zip(u[:40], c[:40])],
              "sha256": {"d3_pairs.py": sha16(Path(__file__).resolve()), "table": sha16(TABLE), "sentences": sha16(src),
                         "pair_counts_c5.npz": sha16(out)},
              "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    summ.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "top_pairs"}))


if __name__ == "__main__":
    main()
