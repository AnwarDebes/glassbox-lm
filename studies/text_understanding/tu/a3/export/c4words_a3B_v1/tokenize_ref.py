#!/usr/bin/env python
"""Reference tokenizer for the C4-word table (c4words_S1000_v1). Turns a text into the table's row ids exactly as
the W1 harness does (glove_feats.glove_tokens, then a lookup by string; a word without a row is dropped).

    python tokenize_ref.py "How do I cancel my card?"      # prints the tokens, their row ids and the dropped words

Rule, in this order:
  1. lower-case the whole text (str.lower);
  2. split contractions as the Stanford tokenizer does: "n't" at a word end -> " n't"; "'s 're 've 'll 'd 'm" at a word
     end -> " 's" etc.;
  3. tokens = every match of  n't | '[a-z]+ | [a-z0-9]+ | [^\\sa-z0-9]   (a run of letters and digits, or one other
     non-space character such as "." "," "?" "-"); nothing else (no stemming, no subwords, no stop-word list);
  4. a token is looked up as a string in words.txt (row i = line i); a token that is not there gets NO vector and is
     dropped (no subword fallback, no [UNK] row).
Only the Python standard library is needed for steps 1-4; numpy reads the vectors.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTR = [(re.compile(r"n't\b"), " n't"), (re.compile(r"'(s|re|ve|ll|d|m)\b"), r" '\1")]
TOK = re.compile(r"n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]")
_ROW = None


def tokens(text: str) -> list[str]:
    t = text.lower()
    for pat, rep in CONTR:
        t = pat.sub(rep, t)
    return TOK.findall(t)


def rows() -> dict:
    global _ROW
    if _ROW is None:
        words = (HERE / "words.txt").read_text(encoding="utf-8").splitlines()
        _ROW = {w: i for i, w in enumerate(words)}
    return _ROW


def ids(text: str) -> list[int]:
    r = rows()
    return [r[w] for w in tokens(text) if w in r]


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "How do I cancel my card? It wasn't me."
    r = rows()
    tk = tokens(text)
    print("tokens :", tk)
    print("row ids:", [r[w] for w in tk if w in r])
    print("dropped (no vector):", [w for w in tk if w not in r])
