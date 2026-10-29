#!/usr/bin/env python
"""Step 3a, the sentences (CPU only; tu/PLAN_3a.md "Text"). Writes tu/a3/c4_sent_train.txt (2,000,000 lines),
tu/a3/c4_sent_held.txt (40,000 lines) and tu/a3/sentences_meta.json; refuses to overwrite.

Pool: the first 2,243,398 documents (S1000's document count) of W1-S's stream (w1/build_corpus_c4_ext.py): shards
00000-00001 in build_corpus_c4.py's default_rng(2026) order without H's last 6,000, then shards 00002-00007 in their
default_rng(2026) order. No potion tokenizer here, so the pool can differ from S1000's end by the few documents W1
cut for an empty potion encoding (declared in PLAN_3a.md). Filter: build_corpus_c4_ext.py's, recomputed the same way
(B.eval_texts(); a sentence is dropped when its key is an evaluation text's key or when it shares an 8-gram of
normalised words with any evaluation text). Sentences: W1's split (B.SENT) of each line. Eligible: 3-64 words by
glove_feats.glove_tokens and at least 3 of them live words of ppw_S1000 (a nonzero row). Exact duplicates (same key)
are kept once. Sampling: documents in a numpy default_rng(20261020) permutation, each eligible new sentence kept with
probability 0.1 from the same generator, until 2,040,000 are kept; the first 40,000 kept are the held-out set.
--smoke DIR: the code path on the first 20,000 documents of shard 00000 only (no stream order), 1,000 held-out and
4,000 training sentences, written to DIR (for the test before the real run; its numbers mean nothing).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(EMB / "w1"))
import build_corpus_c4 as B  # noqa: E402
import glove_feats as GF  # noqa: E402

N_POOL = 2_243_398
N_HELD, N_TRAIN = 40_000, 2_000_000
P_KEEP = 0.1
SEED = 20261020
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> None:
    t0 = time.time()
    smoke = len(sys.argv) == 3 and sys.argv[1] == "--smoke"
    global N_POOL, N_HELD, N_TRAIN
    if smoke:
        N_POOL, N_HELD, N_TRAIN = 20_000, 1_000, 4_000
    out = Path(sys.argv[2]) if smoke else HERE / "a3"
    out.mkdir(exist_ok=True)
    ftr, fho, fm = out / "c4_sent_train.txt", out / "c4_sent_held.txt", out / "sentences_meta.json"
    for f in (ftr, fho, fm):
        assert not f.exists(), f"{f} exists"
    z = np.load(TABLE)
    live = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
    known = {str(w) for w, l in zip(z["words"], live) if l}
    ev, ev_files = B.eval_texts()
    empty = B.key("")
    union_keys, union_grams = set(), set()
    for texts in ev.values():
        ks = {B.key(t) for t in texts}
        ks.discard(empty)
        union_keys |= ks
        for t in texts:
            w = B.norm(t).split()
            for i in range(len(w) - B.NGRAM + 1):
                union_grams.add(" ".join(w[i : i + B.NGRAM]))
    print(f"filter: {len(union_keys):,} keys, {len(union_grams):,} 8-grams ({time.time() - t0:.0f}s)", flush=True)

    def read(shards):
        docs = []
        for s in shards:
            with gzip.open(B.C4 / f"c4-train.{s}-of-01024.json.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line)["text"])
        return docs

    man0 = json.loads((EMB / "w1" / "corpus" / "c4_slices.json").read_text())
    if smoke:
        with gzip.open(B.C4 / "c4-train.00000-of-01024.json.gz", "rt", encoding="utf-8") as f:
            pool = [json.loads(next(f))["text"] for _ in range(N_POOL)]
    else:
        pool = build_pool(read, man0)
    print(f"pool: {len(pool):,} documents ({time.time() - t0:.0f}s)", flush=True)
    sample(pool, rng_seed=SEED, union_keys=union_keys, union_grams=union_grams, known=known, out=(ftr, fho, fm),
           ev_files=ev_files, t0=t0)


def build_pool(read, man0):
    docs01 = read(["00000", "00001"])
    assert len(docs01) == 712635
    order01 = np.random.default_rng(B.SEED).permutation(len(docs01))
    first = [docs01[i] for i in order01[: len(order01) - man0["slices"]["H"]["documents"]]]
    del docs01
    docs_new = read(["00002", "00003", "00004", "00005", "00006", "00007"])
    order_new = np.random.default_rng(B.SEED).permutation(len(docs_new))
    pool = first + [docs_new[int(i)] for i in order_new[: N_POOL - len(first)]]
    assert len(pool) == N_POOL
    return pool


def sample(pool, rng_seed, union_keys, union_grams, known, out, ev_files, t0):
    ftr, fho, fm = out
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(N_POOL)
    st = {"documents_read": 0, "sentences": 0, "key": 0, "ngram": 0, "ineligible": 0, "duplicate": 0, "eligible": 0}
    kept, seen = [], set()
    need = N_HELD + N_TRAIN
    for di in perm:
        st["documents_read"] += 1
        for line in pool[int(di)].split("\n"):
            for s in B.SENT.split(line):
                if not s.strip():
                    continue
                st["sentences"] += 1
                k = B.key(s)
                if k in union_keys:
                    st["key"] += 1
                    continue
                w = B.norm(s).split()
                if any(" ".join(w[i : i + B.NGRAM]) in union_grams for i in range(len(w) - B.NGRAM + 1)):
                    st["ngram"] += 1
                    continue
                toks = GF.glove_tokens(s)
                if not 3 <= len(toks) <= 64 or sum(t in known for t in toks) < 3:
                    st["ineligible"] += 1
                    continue
                if k in seen:
                    st["duplicate"] += 1
                    continue
                st["eligible"] += 1
                if rng.random() < P_KEEP:
                    seen.add(k)
                    kept.append(" ".join(s.split()))
                    if len(kept) == need:
                        break
            if len(kept) == need:
                break
        if len(kept) == need:
            break
    assert len(kept) == need, f"only {len(kept):,} sentences"
    fho.write_text("\n".join(kept[:N_HELD]) + "\n", encoding="utf-8")
    ftr.write_text("\n".join(kept[N_HELD:]) + "\n", encoding="utf-8")
    meta = {"pool_documents": N_POOL, "seed": SEED, "p_keep": P_KEEP, "held": N_HELD, "train": N_TRAIN, "stats": st,
            "filter": {"keys": len(union_keys), "ngrams": len(union_grams), "eval_files": ev_files},
            "live_words": len(known), "sha256": {"c4_sent_train.txt": sha16(ftr), "c4_sent_held.txt": sha16(fho),
                                                 "a3_sentences.py": sha16(Path(__file__).resolve()),
                                                 "build_corpus_c4.py": sha16(EMB / "w1" / "build_corpus_c4.py"),
                                                 "glove_feats.py": sha16(EMB / "glove_feats.py"), "table": sha16(TABLE)},
            "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    fm.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "filter"}))


if __name__ == "__main__":
    main()
