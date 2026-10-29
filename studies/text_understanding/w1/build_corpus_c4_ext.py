#!/usr/bin/env python
"""W1-S: extend the nested C4 slices to S300 and S1000 (rules in LOG.md, W1-S), with a word-token stream for the
vocabulary control.

    python -I build_corpus_c4_ext.py            # writes corpus/c4_S1000.ids.npy, c4_S1000.words.npy, c4_words_vocab.json,
                                                # c4_ext_slices.json (refuses to overwrite)

Order (nested with build_corpus_c4.py's slices): the documents of shards 00000-00001 in build_corpus_c4.py's order
(numpy default_rng(2026) over their 712,635 documents) without H's 6,000 (the last 6,000 of that order), then the
documents of shards 00002-00007 permuted by numpy default_rng(2026). Every document passes the same sentence filter as
build_corpus_c4.py (its key and 8-gram rule, the same evaluation texts, recomputed here) and is tokenized twice:
potion-base-32M ids (as build_corpus_c4.py) and words by glove_feats.glove_tokens (lowercase, Stanford-style
contraction split, regex). A document is cut from both streams if its potion encoding is empty. S300 / S1000 = the
shortest prefixes with at least 300M / 1B potion tokens. Check: the first positions of the new potion stream must equal
c4_S100.ids.npy byte for byte (nesting); the run stops otherwise. Word vocabulary: the 63,091 most frequent word types
of the S1000 documents (ties: alphabetical); other words become -2 in the word stream (out of vocabulary).
CPU only; 2 tokenizer threads.
"""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EMB))
import build_corpus_c4 as B  # noqa: E402
import glove_feats as GF  # noqa: E402

OUT = B.OUT
SIZES = {"S300": 300_000_000, "S1000": 1_000_000_000}
N_WORDS = 63091
CHUNK = 2000


def main() -> None:
    t0 = time.time()
    for f in ("c4_S1000.ids.npy", "c4_S1000.words.npy", "c4_words_vocab.json", "c4_ext_slices.json"):
        assert not (OUT / f).exists(), f"{OUT / f} exists: a new version needs a new name"
    man0 = json.loads((OUT / "c4_slices.json").read_text())
    s100 = np.load(OUT / "c4_S100.ids.npy", mmap_mode="r")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(B.TOK_JSON))

    # ---- the filter of build_corpus_c4.py (same rule, same evaluation texts) ----
    ev, ev_files = B.eval_texts()
    empty = B.key("")
    block, grams = {}, {}
    for name, texts in ev.items():
        ks = {B.key(t) for t in texts}
        ks.discard(empty)
        block[name] = ks
        g = set()
        for t in texts:
            w = B.norm(t).split()
            for i in range(len(w) - B.NGRAM + 1):
                g.add(" ".join(w[i : i + B.NGRAM]))
        grams[name] = g
    union_keys = set().union(*block.values())
    union_grams = set().union(*grams.values())
    drop = {"sentences": 0, "kept_sentences": 0, "key": 0, "ngram": 0}

    def clean(doc: str) -> str:  # build_corpus_c4.py's clean(), without the per-split counts
        lines = []
        for line in doc.split("\n"):
            kept = []
            for s in B.SENT.split(line):
                if not s.strip():
                    continue
                drop["sentences"] += 1
                if B.key(s) in union_keys:
                    drop["key"] += 1
                    continue
                w = B.norm(s).split()
                if any(" ".join(w[i : i + B.NGRAM]) in union_grams for i in range(len(w) - B.NGRAM + 1)):
                    drop["ngram"] += 1
                    continue
                kept.append(s)
                drop["kept_sentences"] += 1
            if kept:
                lines.append(" ".join(kept))
        return "\n".join(lines)

    # ---- documents and order ----
    def read(shards):
        docs = []
        for s in shards:
            with gzip.open(B.C4 / f"c4-train.{s}-of-01024.json.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line)["text"])
        return docs

    docs01 = read(["00000", "00001"])
    order01 = np.random.default_rng(B.SEED).permutation(len(docs01))
    assert len(docs01) == 712635
    h_docs = man0["slices"]["H"]["documents"]
    first = [docs01[i] for i in order01[: len(order01) - h_docs]]
    del docs01
    new_shards = ["00002", "00003", "00004", "00005", "00006", "00007"]
    docs_new = read(new_shards)
    order_new = np.random.default_rng(B.SEED).permutation(len(docs_new))
    print(f"documents: {len(first):,} from shards 00000-00001 (H left out), {len(docs_new):,} new ({time.time() - t0:.0f}s)", flush=True)

    def stream():
        for d in first:
            yield d
        for i in order_new:
            yield docs_new[int(i)]

    word_id: dict[str, int] = {}
    ids_parts, w_parts, n_tok, n_docs, n_empty, length, ends = [], [], 0, 0, 0, 0, {}
    checked = False
    end100 = man0["slices"]["S100"]["end_offset_in_S100"]
    buf = []

    def flush(texts):
        nonlocal n_tok, n_docs, n_empty, length
        cl = [clean(t) for t in texts]
        enc = tok.encode_batch(cl, add_special_tokens=False)
        for c, e in zip(cl, enc):
            if len(e.ids) == 0:
                n_empty += 1
                continue
            ids_parts.append(np.asarray(e.ids + [-1], dtype=np.int32))
            ws = GF.glove_tokens(c)
            w_parts.append(np.asarray([word_id.setdefault(w, len(word_id)) for w in ws] + [-1], dtype=np.int32))
            length += len(e.ids) + 1
            n_tok += len(e.ids)
            n_docs += 1
            for name, size in SIZES.items():
                if name not in ends and n_tok >= size:
                    ends[name] = {"end_offset": length, "documents": n_docs, "tokens": n_tok,
                                  "word_end_offset": sum(len(a) for a in w_parts)}

    for d in stream():
        buf.append(d)
        if len(buf) == CHUNK:
            flush(buf)
            buf = []
            if not checked and length >= end100:
                head = np.concatenate(ids_parts)[:end100]
                assert np.array_equal(head, np.asarray(s100[:end100])), "NESTING CHECK FAILED: the new stream does not start with S100"
                checked = True
                print(f"nesting check passed at {n_docs:,} documents ({time.time() - t0:.0f}s)", flush=True)
            if len(ends) == len(SIZES):
                break
    assert checked and len(ends) == len(SIZES), f"only {n_tok:,} tokens: more shards needed"
    ids = np.concatenate(ids_parts)[: ends["S1000"]["end_offset"]]
    words_prov = np.concatenate(w_parts)[: ends["S1000"]["word_end_offset"]]
    del ids_parts, w_parts
    # word vocabulary: the N_WORDS most frequent types of the S1000 documents (ties alphabetical)
    inv = np.empty(len(word_id), dtype=object)
    for w, i in word_id.items():
        inv[i] = w
    cnt = np.bincount(words_prov[words_prov >= 0], minlength=len(word_id))
    order_w = sorted(range(len(word_id)), key=lambda i: (-cnt[i], inv[i]))[:N_WORDS]
    remap = np.full(len(word_id), -2, dtype=np.int32)
    remap[np.asarray(order_w)] = np.arange(len(order_w), dtype=np.int32)
    words = np.where(words_prov >= 0, remap[np.maximum(words_prov, 0)], -1).astype(np.int32)
    np.save(OUT / "c4_S1000.ids.npy", ids)
    np.save(OUT / "c4_S1000.words.npy", words)
    (OUT / "c4_words_vocab.json").write_text(json.dumps({"words": [inv[i] for i in order_w], "counts": [int(cnt[i]) for i in order_w]}))
    sh = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype=np.int32).tobytes()).hexdigest()  # noqa: E731
    slices = {}
    for name, e in ends.items():
        a = ids[: e["end_offset"]]
        slices[name] = {"end_offset": int(e["end_offset"]), "documents": int(e["documents"]), "tokens": int(e["tokens"]),
                        "sha256_int32": sh(a), "word_end_offset": int(e["word_end_offset"])}
    wv = words
    man = {"written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": str(Path(__file__).resolve()), "sha256": B.sha256_file(Path(__file__).resolve())},
           "build_corpus_c4_sha256": B.sha256_file(HERE / "build_corpus_c4.py"),
           "inputs": {"shards": {p: B.sha256_file(B.C4 / f"c4-train.{p}-of-01024.json.gz") for p in ["00000", "00001"] + new_shards},
                      "c4_S100_sha256": B.sha256_file(OUT / "c4_S100.ids.npy")},
           "order": "shards 00000-00001 in build_corpus_c4.py's order without H, then shards 00002-00007 by default_rng(2026)",
           "nesting_check": "the first S100 positions equal c4_S100.ids.npy byte for byte: passed",
           "filter_counts_beyond_shards_reused": drop, "documents_empty_after_filter": int(n_empty),
           "slices": slices,
           "words": {"vocabulary": len(order_w), "types_seen": len(word_id), "tokens_in_S1000": int((wv != -1).sum()),
                     "oov_tokens_in_S1000": int((wv == -2).sum()), "min_count_in_vocab": int(cnt[order_w[-1]]),
                     "sha256_int32": sh(wv)},
           "files": {f: B.sha256_file(OUT / f) for f in ("c4_S1000.ids.npy", "c4_S1000.words.npy", "c4_words_vocab.json")},
           "seconds": round(time.time() - t0, 1)}
    (OUT / "c4_ext_slices.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"slices": slices, "words": man["words"], "filter": drop, "seconds": man["seconds"]}))


if __name__ == "__main__":
    main()
