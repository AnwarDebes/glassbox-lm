"""KenLM reference models (lmplz, build_binary, query) on the tally splits.

Token mapping, the same as the week-0 check of the reference toolkit (sentence mode):
- one sentence per line; the <eos> of the tally streams is not written, KenLM appends </s> to
  every line and predicts it, so </s> is scored where the study's own models score <eos>;
- KenLM puts <s> in front of every line as context; it is never predicted (the study's BOS);
- the literal PTB word <unk> is renamed UNKPTB, an ordinary word (lmplz refuses <unk> in the
  corpus, and with --skip_symbols it would delete it);
- KenLM's own <unk> is its out-of-vocabulary class. It receives only vocabulary words that do
  not occur in the reduced train text; its probability is the uniform share of the unigram level.
lmplz runs with its defaults: modified Kneser-Ney, interpolated unigrams, no pruning.

The binaries are found in tools/kenlm/bin of the workspace (three levels above the study folder),
or in the folder named by the environment variable KENLM_BIN. info() (added 2026-10-25) records
what can be read about them; KenLM has no version number and the binaries hold no version string.
"""
import hashlib
import math
import os
import re
import subprocess

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BIN = os.environ.get("KENLM_BIN") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(STUDY))), "tools", "kenlm", "bin")
UNK_SUB = "UNKPTB"


def info():
    """What can be read about the binaries in BIN: the folder, the sha256 of lmplz, build_binary
    and query, and the maximum order that query prints with its usage text. The commit the
    binaries were built from is not in them; when the README of the tools folder (two levels
    above BIN) states a 40-digit commit, it is copied with the path it was read from."""
    out = {"bin": BIN, "sha256": {}, "max_order": None, "source_commit_stated": None, "stated_in": None}
    for b in ("lmplz", "build_binary", "query"):
        p = os.path.join(BIN, b)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out["sha256"][b] = hashlib.sha256(f.read()).hexdigest()
    try:
        r = subprocess.run([os.path.join(BIN, "query")], capture_output=True, text=True, timeout=60)
        m = re.search(r"compiled with maximum order (\d+)", r.stdout + r.stderr)
        out["max_order"] = int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        pass
    readme = os.path.join(os.path.dirname(os.path.dirname(BIN)), "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf8") as f:
            m = re.search(r"commit `([0-9a-f]{40})`", f.read())
        if m:
            out["source_commit_stated"], out["stated_in"] = m.group(1), readme
    return out


def write_text(stream, vocab, eos, path):
    """One sentence per line, <eos> dropped, <unk> renamed. Returns the number of lines."""
    words = np.array([UNK_SUB if w == "<unk>" else w for w in vocab], dtype=object)
    assert UNK_SUB not in vocab
    ends = np.flatnonzero(np.asarray(stream) == eos)
    assert len(ends) and ends[-1] == len(stream) - 1, "stream must end with <eos>"
    with open(path, "w") as f:
        start = 0
        for e in ends:
            f.write(" ".join(words[stream[start:e]]) + "\n")
            start = e + 1
    return len(ends)


def lmplz(order, text, arpa, tmp, memory="1G"):
    """Build an ARPA model. Returns the discounts lmplz printed, {order: [D1, D2, D3+]}."""
    os.makedirs(tmp, exist_ok=True)
    with open(text) as fi:
        r = subprocess.run([os.path.join(BIN, "lmplz"), "-o", str(order), "-S", memory, "-T", tmp,
                            "--arpa", arpa], stdin=fi, capture_output=True, text=True, check=True)
    with open(arpa + ".log", "w") as f:
        f.write(r.stderr)
    disc = {}
    for m in re.finditer(r"^(\d+) (\d+) D1=(\S+) D2=(\S+) D3\+=(\S+)$", r.stderr, re.M):
        disc[int(m.group(1))] = [float(m.group(i)) for i in (3, 4, 5)]
    return disc


def build_binary(arpa, binary):
    subprocess.run([os.path.join(BIN, "build_binary"), "-s", arpa, binary], capture_output=True,
                   check=True)


def query(binary, text):
    """Natural log probability of every token of every line, </s> included, in text order.

    Returns (logp, oov) as arrays; oov marks tokens that KenLM mapped to its <unk>."""
    with open(text) as fi:
        r = subprocess.run([os.path.join(BIN, "query"), "-v", "word", "-l", "read", binary], stdin=fi,
                           capture_output=True, text=True, check=True)
    lp, oov = [], []
    ln10 = math.log(10)
    item = re.compile(r"^(.*)=(\d+) (\d+) (\S+)$")
    for line in r.stdout.split("\n"):
        for it in line.split("\t"):
            m = item.match(it)
            if m:
                lp.append(float(m.group(4)) * ln10)
                oov.append(m.group(2) == "0")
    return np.array(lp), np.array(oov)
