#!/usr/bin/env python
"""Overlap of the C4 S1000 slice (behind ppw_S1000 and the c4words-S1000-v1 table) with MultiWOZ 2.2 dev and test
utterance text, by W1's corpus dedupe rule. A measurement for the decision_layer R24 (decided 21:20Z); no table is rebuilt.

The rule (w1/build_corpus_c4.py): text is split into lines and each line into sentences (after . ! or ? followed by
white space); key(t) = sha256 of norm(t), norm = NFKC + lowercase + the words [a-z0-9]+('[a-z0-9]+)* joined by one
space (curly apostrophes mapped to '). A C4 sentence matches an utterance VERBATIM when the two keys are equal, and by
8-GRAM when it holds 8 consecutive words that also occur, in that order, in an utterance of 8 or more words.
The slice: build_corpus_c4_ext.py's document order (shards 00000-00001 in build_corpus_c4.py's order without H's
documents, then shards 00002-00007 by numpy default_rng(2026)), each document cleaned by W1's filter against W1's
evaluation texts exactly as when the slice was built, and kept when the cleaned text is not empty, up to the
manifest's 2,243,398 documents of S1000. (The build also cut a document whose potion encoding was empty; this replay
has no tokenizer, so such a rare document would be counted here; the end can differ by a few documents.)
MultiWOZ: data/untrusted/multiwoz22/{dev,test}/dialogues_*.json of the decision-layer study, read as JSON; only the
"utterance" text field of each turn (no labels, no acts). CPU, one thread. Writes mwoz_leak/overlap.json and
mwoz_leak/overlap.txt (refuses to overwrite).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import glob  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB / "w1"))
import build_corpus_c4 as B  # noqa: E402

MW = Path(f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/multiwoz22")
MAN = EMB / "w1" / "corpus" / "c4_ext_slices.json"
MAN0 = EMB / "w1" / "corpus" / "c4_slices.json"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    t0 = time.time()
    out_json, out_txt = HERE / "overlap.json", HERE / "overlap.txt"
    assert not out_json.exists() and not out_txt.exists(), "outputs exist; not overwritten"
    # MultiWOZ utterances (text only)
    utt, files = [], {}
    for split in ("dev", "test"):
        for f in sorted(glob.glob(str(MW / split / "dialogues_*.json"))):
            files[f"{split}/{Path(f).name}"] = sha16(Path(f))
            for dlg in json.loads(Path(f).read_text()):
                for turn in dlg["turns"]:
                    utt.append((split, turn["utterance"]))
    keys = defaultdict(list)
    grams = defaultdict(set)
    nwords = []
    for i, (_, t) in enumerate(utt):
        w = B.norm(t).split()
        nwords.append(len(w))
        if w:
            keys[B.key(t)].append(i)
        for j in range(len(w) - B.NGRAM + 1):
            grams[" ".join(w[j : j + B.NGRAM])].add(i)
    print(f"utterances {len(utt):,}; distinct keys {len(keys):,}; distinct 8-grams {len(grams):,}", flush=True)
    # W1's filter, as built
    ev, _ = B.eval_texts()
    empty = B.key("")
    union_keys, union_grams = set(), set()
    for texts in ev.values():
        for t in texts:
            k = B.key(t)
            if k != empty:
                union_keys.add(k)
            w = B.norm(t).split()
            for j in range(len(w) - B.NGRAM + 1):
                union_grams.add(" ".join(w[j : j + B.NGRAM]))
    verb, ngram = defaultdict(int), defaultdict(int)
    example = {}

    def clean_and_match(doc: str) -> bool:
        any_kept = False
        for line in doc.split("\n"):
            for s in B.SENT.split(line):
                if not s.strip():
                    continue
                k = B.key(s)
                if k in union_keys:
                    continue
                w = B.norm(s).split()
                g = [" ".join(w[j : j + B.NGRAM]) for j in range(len(w) - B.NGRAM + 1)]
                if any(x in union_grams for x in g):
                    continue
                any_kept = True
                if k in keys:
                    for i in keys[k]:
                        verb[i] += 1
                        example.setdefault(i, ("verbatim", s[:300]))
                for x in g:
                    for i in grams.get(x, ()):
                        ngram[i] += 1
                        example.setdefault(i, ("8-gram", s[:300]))
        return any_kept

    man, man0 = json.loads(MAN.read_text()), json.loads(MAN0.read_text())
    target = man["slices"]["S1000"]["documents"]

    def read(shards):
        docs = []
        for s in shards:
            with gzip.open(B.C4 / f"c4-train.{s}-of-01024.json.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line)["text"])
        return docs

    kept = 0
    docs01 = read(["00000", "00001"])
    assert len(docs01) == 712635
    order01 = np.random.default_rng(B.SEED).permutation(len(docs01))
    h_docs = man0["slices"]["H"]["documents"]
    for i in order01[: len(order01) - h_docs]:
        if kept >= target:
            break
        kept += clean_and_match(docs01[int(i)])
    del docs01
    print(f"after shards 00000-00001: {kept:,} documents ({time.time() - t0:.0f}s)", flush=True)
    if kept < target:
        docs_new = read(["00002", "00003", "00004", "00005", "00006", "00007"])
        order_new = np.random.default_rng(B.SEED).permutation(len(docs_new))
        for n, i in enumerate(order_new):
            if kept >= target:
                break
            kept += clean_and_match(docs_new[int(i)])
            if n % 200000 == 0:
                print(f"  {kept:,} documents ({time.time() - t0:.0f}s)", flush=True)
    res = {"rule": "W1's dedupe rule (verbatim key or a shared 8-gram), C4 S1000 sentences vs MultiWOZ 2.2 utterances",
           "documents_replayed": kept, "documents_in_manifest": target}
    lines = []
    for split in ("dev", "test"):
        idx = [i for i, (s, _) in enumerate(utt) if s == split]
        v = [i for i in idx if verb[i]]
        g = [i for i in idx if ngram[i]]
        e = [i for i in idx if verb[i] or ngram[i]]
        long_v = [i for i in v if nwords[i] >= 8]
        res[split] = {"utterances": len(idx), "verbatim": len(v), "eight_gram": len(g), "either": len(e),
                      "share_either": round(len(e) / len(idx), 5), "verbatim_of_8_or_more_words": len(long_v),
                      "utterances_of_8_or_more_words": sum(1 for i in idx if nwords[i] >= 8)}
        lines.append(f"{split}: {len(idx):,} utterances; verbatim {len(v):,}; 8-gram {len(g):,}; either {len(e):,} "
                     f"({100 * len(e) / len(idx):.2f}%); verbatim matches of whole utterances with 8+ words: {len(long_v):,}")
    ex = sorted(i for i in example)[:: max(1, len(example) // 10)][:10]
    lines.append("examples (utterance | how | C4 sentence, first 300 characters):")
    for i in ex:
        how, s = example[i]
        lines.append(f"  [{utt[i][0]}] {utt[i][1]!r} | {how} | {s!r}")
    res["examples"] = [{"split": utt[i][0], "utterance": utt[i][1], "how": example[i][0], "c4": example[i][1]} for i in ex]
    res["sha256"] = {"script": sha16(Path(__file__).resolve()), "c4_ext_slices.json": sha16(MAN), **files}
    res["seconds"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(res, indent=1))
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
