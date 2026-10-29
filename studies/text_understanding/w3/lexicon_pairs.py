#!/usr/bin/env python
"""W3 step 1 (EXPLORATORY, intent VAL only; rules in LOG.md before any run): synonym and antonym pairs from Open
English WordNet 2025 among the C4 words of ppw_S1000. CPU only.

Input: the WNDB plain-text data files of data_untrusted/oewn2025/wndb/oewn2025 (data.noun / .verb / .adj / .adv),
read as text (lines starting with a space are the licence header). Per line: offset, lexicographer file, synset
type, the word count (hex), the words with their lex ids, the pointer count, the pointers (symbol, offset, part of
speech, source/target as 4 hex digits). A lemma is lowercased and an adjective marker "(a)", "(p)" or "(ip)" is
removed; it counts only when it is a single C4 word: letters, digits and apostrophes only, and a live row of
ppw_S1000 (w1/tables/ppw_S1000_d500_e0.5.npz).
- Synonym pairs: every unordered pair of such lemmas in one synset (deduplicated over synsets).
- Antonym pairs: the lexical "!" pointers whose source and target lemmas are such words.
- A pair in both sets is dropped from both.
- Control graphs: a configuration-model shuffle of each set's endpoints (numpy default_rng(0) for synonyms,
  default_rng(1) for antonyms): the same words with the same degrees, pairs re-drawn; self pairs and repeats dropped.
Writes w3/pairs/oewn_pairs.npz (index pairs into the ppw_S1000 rows) and w3/pairs/oewn_pairs.json (counts, hashes);
refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
WNDB = EMB / "data_untrusted" / "oewn2025" / "wndb" / "oewn2025"
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
FILES = {"n": "data.noun", "v": "data.verb", "a": "data.adj", "r": "data.adv"}
WORD = re.compile(r"^[a-z0-9']+$")
MARK = re.compile(r"\((a|p|ip)\)$")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def parse():
    words, ptrs, n_syn = {}, [], 0
    for pf, fn in FILES.items():
        with open(WNDB / fn, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(" ") or not line.strip():
                    continue
                f = line.split(" | ", 1)[0].split()
                off, wcnt, i, ws = f[0], int(f[3], 16), 4, []
                for _ in range(wcnt):
                    ws.append(MARK.sub("", f[i].lower()))
                    i += 2
                pcnt = int(f[i])
                i += 1
                for _ in range(pcnt):
                    sym, toff, tpos, st = f[i : i + 4]
                    i += 4
                    if sym == "!" and st != "0000":
                        ptrs.append((pf, off, int(st[:2], 16) - 1, "a" if tpos in ("a", "s") else tpos, toff, int(st[2:], 16) - 1))
                words[(pf, off)] = ws
                n_syn += 1
    return words, ptrs, n_syn


def shuffle_graph(pairs: np.ndarray, seed: int) -> np.ndarray:
    stubs = pairs.reshape(-1).copy()
    np.random.default_rng(seed).shuffle(stubs)
    p = stubs.reshape(-1, 2)
    p = np.sort(p, axis=1)
    p = p[p[:, 0] != p[:, 1]]
    return np.unique(p, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "pairs")  # a test copy elsewhere
    out_dir = ap.parse_args().out_dir
    if (out_dir / "oewn_pairs.npz").exists():
        sys.exit(f"{out_dir}/oewn_pairs.npz exists; not overwritten")
    z = np.load(TABLE)
    tw = [str(w) for w in z["words"]]
    live = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
    index = {w: i for i, w in enumerate(tw) if live[i]}
    words, ptrs, n_syn = parse()
    ok = lambda w: WORD.match(w) is not None and w in index  # noqa: E731
    syn = set()
    for ws in words.values():
        ids = sorted({index[w] for w in ws if ok(w)})
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                syn.add((ids[a], ids[b]))
    ant = set()
    miss = 0
    for pf, off, s, tpf, toff, t in ptrs:
        src, tgt = words.get((pf, off)), words.get((tpf, toff))
        if src is None or tgt is None or s >= len(src) or t >= len(tgt):
            miss += 1
            continue
        if ok(src[s]) and ok(tgt[t]) and src[s] != tgt[t]:
            a, b = sorted((index[src[s]], index[tgt[t]]))
            ant.add((a, b))
    both = syn & ant
    syn -= both
    ant -= both
    S = np.array(sorted(syn), dtype=np.int64).reshape(-1, 2)
    A = np.array(sorted(ant), dtype=np.int64).reshape(-1, 2)
    Sr, Ar = shuffle_graph(S, 0), shuffle_graph(A, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "oewn_pairs.npz", syn=S, ant=A, syn_random=Sr, ant_random=Ar)
    info = {"synsets": n_syn, "antonym_pointers": len(ptrs), "pointers_unresolved": miss,
            "synonym_pairs": int(S.shape[0]), "antonym_pairs": int(A.shape[0]), "dropped_in_both": len(both),
            "synonym_words": int(np.unique(S).shape[0]), "antonym_words": int(np.unique(A).shape[0]),
            "random_synonym_pairs": int(Sr.shape[0]), "random_antonym_pairs": int(Ar.shape[0]),
            "c4_live_words": int(live.sum()),
            "examples_synonyms": [[tw[a], tw[b]] for a, b in S[np.random.default_rng(5).choice(len(S), 12, replace=False)]],
            "examples_antonyms": [[tw[a], tw[b]] for a, b in A[np.random.default_rng(6).choice(len(A), 12, replace=False)]],
            "sha256": {"lexicon_pairs.py": sha16(Path(__file__).resolve()), "ppw_S1000_d500_e0.5.npz": sha16(TABLE),
                       **{fn: sha16(WNDB / fn) for fn in FILES.values()}}}
    (out_dir / "oewn_pairs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
