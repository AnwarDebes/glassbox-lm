#!/usr/bin/env python
"""W2 H-B step 0, part 1 (W2_NOTE.md; rules in LOG.md before the run): count the question -> answer pairs inside the
documents of the C4 slice S1000 that the C4-word table was counted on, and draw the hand-check samples.

    python -I pairs_scan.py --selftest     # the pair rule on synthetic lines only; reads no C4 file
    python -I pairs_scan.py                # the scan; writes w2/corpus/ (refuses to overwrite)

S1000, recomputed exactly as w1/build_corpus_c4_ext.py: the documents of shards 00000-00001 in numpy
default_rng(2026) order without H's 6,000 (the last 6,000 of that order), then the documents of shards 00002-00007
permuted by numpy default_rng(2026); every document passes build_corpus_c4.py's sentence filter (its key rule and
8-gram rule against every evaluation split, the same evaluation texts); a document whose potion encoding is empty is
skipped; S1000 = the shortest prefix with at least 1B potion tokens. Check: the prefix must hold exactly the
documents and tokens of w1/corpus/c4_ext_slices.json (2,243,398 / 1,000,000,436), else the scan stops and keeps
nothing. Pairs are read from the FILTERED text, so no sentence that matches an evaluation text is in any pair.
Pair rule (fixed before the run), on the lines of a filtered document:
  - a leading "Q:", "Q.", "Q)", "Question:" (or "A:", "Answer:", "Ans:" on an answer) is removed;
  - a question line ends with "?" and has 3-40 words (white-space split) and at least one letter;
  - its answer is the next line if that line does not end with "?" and has at least 5 words; kept to its first 100
    words;
  - a document is FAQ-like when it holds at least 3 pairs; url_faq when its url contains "faq" (any case).
Question <-> question pairs: questions whose answers are equal after build_corpus_c4.norm, with at least 2 distinct
normalised questions; all pairs inside a group of at most its first 10 questions.
Hand-check samples (numpy default_rng(2026)): 200 pairs of FAQ-like documents and 100 question <-> question pairs,
each side cut to 30 words, in corpus/handcheck_*.txt.
Output (w2/corpus/, local only; raw pair text is never redistributed): pairs_S1000.jsonl.gz (q, a, doc position in
the S1000 order, faq_like, url_faq), qq_S1000.jsonl.gz, pairs_scan.json (counts, timings, input and script hashes),
handcheck_faq_200.txt, handcheck_qq_100.txt. CPU only, one tokenizer thread.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # .../studies/text_understanding/w2
EMB = HERE.parent
W1 = EMB / "w1"
OUT = HERE / "corpus"
sys.path.insert(0, str(W1))
sys.path.insert(0, str(EMB))
import build_corpus_c4 as B  # noqa: E402  (sets RAYON 2 / OMP 1 / no CUDA at import; RAYON is lowered below)

os.environ["RAYON_NUM_THREADS"] = "1"
S1000_DOCS, S1000_TOKENS, S1000_SIZE = 2_243_398, 1_000_000_436, 1_000_000_000
CHUNK = 2000
QPRE = re.compile(r"^\s*(?:q|question)\s*[:.)]\s*", re.I)
APRE = re.compile(r"^\s*(?:a|answer|ans)\s*[:.)]\s*", re.I)
LETTER = re.compile(r"[A-Za-z]")


def pairs_of(text: str) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in text.split("\n")]
    out = []
    for i in range(len(lines) - 1):
        q = QPRE.sub("", lines[i]).strip()
        if not q.endswith("?") or not LETTER.search(q) or not 3 <= len(q.split()) <= 40:
            continue
        a = APRE.sub("", lines[i + 1]).strip()
        if a.endswith("?") or len(a.split()) < 5:
            continue
        out.append((q, " ".join(a.split()[:100])))
    return out


def selftest() -> None:
    doc = "\n".join([
        "Frequently asked questions",
        "Q: How do I reset my password?",
        "A: Go to the login page and click on the link that says forgot password.",
        "What is your refund policy?",
        "Is it free?",
        "Yes, it is free for the first thirty days of use.",
        "Why?",
        "Because we said so and that is the end of the story here.",
        "Can I change my delivery address after I have placed the order?",
        "Yes. Write to us within two hours and we will change it for you.",
    ])
    got = pairs_of(doc)
    for q, a in got:
        print(f"Q [{q}] -> A [{a}]")
    want = [("How do I reset my password?", "Go to the login page and click on the link that says forgot password."),
            ("Is it free?", "Yes, it is free for the first thirty days of use."),
            ("Can I change my delivery address after I have placed the order?",
             "Yes. Write to us within two hours and we will change it for you.")]
    assert got == want, "selftest FAILED"
    print("selftest passed: 3 pairs; 'What is your refund policy?' (next line a question) and 'Why?' (1 word) left out")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cut(s: str, n: int = 30) -> str:
    w = s.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    t0 = time.time()
    names = ("pairs_S1000.jsonl.gz", "qq_S1000.jsonl.gz", "pairs_scan.json", "handcheck_faq_200.txt", "handcheck_qq_100.txt")
    for f in names:
        assert not (OUT / f).exists(), f"{OUT / f} exists: a new version needs a new name"
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(B.TOK_JSON))
    man0 = json.loads((W1 / "corpus" / "c4_slices.json").read_text())

    # ---- the filter of build_corpus_c4.py (same rule, same evaluation texts), as w1/build_corpus_c4_ext.py ----
    ev, ev_files = B.eval_texts()
    empty = B.key("")
    union_keys, union_grams = set(), set()
    for texts in ev.values():
        for t in texts:
            k = B.key(t)
            if k != empty:
                union_keys.add(k)
            w = B.norm(t).split()
            for i in range(len(w) - B.NGRAM + 1):
                union_grams.add(" ".join(w[i : i + B.NGRAM]))

    def clean(doc: str) -> str:
        lines = []
        for line in doc.split("\n"):
            kept = []
            for s in B.SENT.split(line):
                if not s.strip() or B.key(s) in union_keys:
                    continue
                w = B.norm(s).split()
                if any(" ".join(w[i : i + B.NGRAM]) in union_grams for i in range(len(w) - B.NGRAM + 1)):
                    continue
                kept.append(s)
            if kept:
                lines.append(" ".join(kept))
        return "\n".join(lines)

    # ---- documents and order (w1/build_corpus_c4_ext.py) ----
    def read(shards):
        docs = []
        for s in shards:
            with gzip.open(B.C4 / f"c4-train.{s}-of-01024.json.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    docs.append((r["text"], r.get("url", "")))
        return docs

    docs01 = read(["00000", "00001"])
    assert len(docs01) == 712635
    order01 = np.random.default_rng(B.SEED).permutation(len(docs01))
    h_docs = man0["slices"]["H"]["documents"]
    first = [docs01[i] for i in order01[: len(order01) - h_docs]]
    del docs01
    docs_new = read(["00002", "00003", "00004", "00005", "00006", "00007"])
    order_new = np.random.default_rng(B.SEED).permutation(len(docs_new))
    print(f"documents read ({time.time() - t0:.0f}s)", flush=True)

    def stream():
        yield from first
        for i in order_new:
            yield docs_new[int(i)]

    pairs, n_docs, n_tok, n_faq_docs, n_url_docs, done = [], 0, 0, 0, 0, False
    buf = []

    def flush(items):
        nonlocal n_docs, n_tok, n_faq_docs, n_url_docs, done
        cl = [clean(t) for t, _ in items]
        enc = tok.encode_batch(cl, add_special_tokens=False)
        for (_, url), c, e in zip(items, cl, enc):
            if done:
                return
            if len(e.ids) == 0:
                continue
            n_docs += 1
            n_tok += len(e.ids)
            ps = pairs_of(c)
            faq_like, url_faq = len(ps) >= 3, "faq" in (url or "").lower()
            n_faq_docs += faq_like
            n_url_docs += url_faq
            for q, a in ps:
                pairs.append({"q": q, "a": a, "doc": n_docs - 1, "faq_like": faq_like, "url_faq": url_faq})
            if n_tok >= S1000_SIZE:
                done = True

    for d in stream():
        buf.append(d)
        if len(buf) == CHUNK:
            flush(buf)
            buf = []
            if done:
                break
            if n_docs % 200_000 < CHUNK:
                print(f"{n_docs:,} documents, {n_tok:,} tokens, {len(pairs):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    if not done and buf:
        flush(buf)
    if (n_docs, n_tok) != (S1000_DOCS, S1000_TOKENS):
        sys.exit(f"S1000 CHECK FAILED: {n_docs:,} documents / {n_tok:,} tokens, expected {S1000_DOCS:,} / {S1000_TOKENS:,}; nothing kept")
    print(f"S1000 check passed: {n_docs:,} documents / {n_tok:,} tokens ({time.time() - t0:.0f}s)", flush=True)

    # ---- question <-> question pairs: questions sharing an answer ----
    groups: dict[str, list[str]] = {}
    for p in pairs:
        groups.setdefault(B.norm(p["a"]), []).append(p["q"])
    qq = []
    n_groups = 0
    for a_norm, qs in groups.items():
        uniq, seen = [], set()
        for q in qs:
            k = B.norm(q)
            if k and k not in seen:
                seen.add(k)
                uniq.append(q)
        if len(uniq) < 2 or not a_norm:
            continue
        n_groups += 1
        uniq = uniq[:10]
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                qq.append({"q1": uniq[i], "q2": uniq[j], "answer_sha16": hashlib.sha256(a_norm.encode()).hexdigest()[:16]})

    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "pairs_S1000.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    with gzip.open(OUT / "qq_S1000.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in qq:
            f.write(json.dumps(p) + "\n")
    rng = np.random.default_rng(B.SEED)
    faq = [p for p in pairs if p["faq_like"]]
    pick = rng.choice(len(faq), size=min(200, len(faq)), replace=False) if faq else []
    with open(OUT / "handcheck_faq_200.txt", "w", encoding="utf-8") as f:
        f.write("# 200 question -> answer pairs of FAQ-like S1000 documents (numpy default_rng(2026)); local only\n")
        for n, i in enumerate(sorted(int(x) for x in pick), 1):
            f.write(f"{n}. Q: {cut(faq[i]['q'])}\n   A: {cut(faq[i]['a'])}\n")
    pick2 = rng.choice(len(qq), size=min(100, len(qq)), replace=False) if qq else []
    with open(OUT / "handcheck_qq_100.txt", "w", encoding="utf-8") as f:
        f.write("# 100 question <-> question pairs (questions sharing an answer; numpy default_rng(2026)); local only\n")
        for n, i in enumerate(sorted(int(x) for x in pick2), 1):
            f.write(f"{n}. Q1: {cut(qq[i]['q1'])}\n   Q2: {cut(qq[i]['q2'])}\n")
    info = {
        "step": "W2 H-B step 0 part 1 (exploratory; no evaluation data read except as the filter's blocklist)",
        "s1000": {"documents": n_docs, "tokens": n_tok, "check": "equal to w1/corpus/c4_ext_slices.json"},
        "pairs": len(pairs), "pairs_in_faq_like_docs": len(faq), "pairs_in_url_faq_docs": sum(p["url_faq"] for p in pairs),
        "faq_like_docs": n_faq_docs, "url_faq_docs": n_url_docs,
        "qq_groups": n_groups, "qq_pairs": len(qq),
        "pairs_per_shard_equivalent": {"note": "S1000 = 2,243,398 documents of 2,843,538 in 8 shards (H left out)",
                                       "faq_like_pairs_per_1000_docs": round(1000 * len(faq) / n_docs, 3)},
        "inputs": {"c4_ext_slices.json": sha16(W1 / "corpus" / "c4_ext_slices.json"),
                   "tokenizer.json": sha16(B.TOK_JSON),
                   "shards": {s: sha16(B.C4 / f"c4-train.{s}-of-01024.json.gz") for s in
                              ("00000", "00001", "00002", "00003", "00004", "00005", "00006", "00007")},
                   "eval_files": ev_files},
        "sha256": {"pairs_scan.py": sha16(Path(__file__).resolve()), "build_corpus_c4.py": sha16(W1 / "build_corpus_c4.py")},
        "outputs": {f: sha16(OUT / f) for f in ("pairs_S1000.jsonl.gz", "qq_S1000.jsonl.gz", "handcheck_faq_200.txt", "handcheck_qq_100.txt")},
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - t0, 1),
    }
    (OUT / "pairs_scan.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("inputs",)}), flush=True)


if __name__ == "__main__":
    main()
