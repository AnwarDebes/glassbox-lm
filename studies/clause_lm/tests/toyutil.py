"""Shared helpers of the clause_lm tests (not a test itself; imported by the test scripts).

- toy_corpus: sentences from a fixed random bigram chain over a small vocabulary, so contexts repeat;
- write_ptb: writes such sentences in the layout that src/ptb.py reads (one sentence per line);
- use_src: puts src/ on sys.path and points ptb.PTB and runlog.ROOT at a temporary folder, so that no test reads the
  Penn Treebank text or writes into results/;
- load_module_copy: imports a copy of one src/ file (for example the version before a fix) beside the current
  src/runlog.py and src/ptb.py, under a fresh module name;
- Tripwire: records every file a block opens whose path contains a given part (used to show that the test split is
  never opened).
"""

import builtins
import importlib.util
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SRC = STUDY / "src"


def toy_corpus(seed=0, n_sent=300, n_words=30, min_len=3, max_len=9):
    """Sentences (lists of words) from a random bigram chain; word i is 'w<i>'."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(n_words, 0.3), size=n_words)
    start = rng.dirichlet(np.full(n_words, 0.5))
    out = []
    for _ in range(n_sent):
        n = int(rng.integers(min_len, max_len + 1))
        w = int(rng.choice(n_words, p=start))
        sent = [w]
        for _ in range(n - 1):
            w = int(rng.choice(n_words, p=trans[w]))
            sent.append(w)
        out.append([f"w{i}" for i in sent])
    return out


def write_ptb(root, train, valid):
    """root/ptb/{train,validation}/text.utf8; returns root/ptb. No test file is written."""
    ptb_dir = Path(root) / "ptb"
    for split, sents in (("train", train), ("validation", valid)):
        (ptb_dir / split).mkdir(parents=True, exist_ok=True)
        (ptb_dir / split / "text.utf8").write_text("".join(" ".join(s) + "\n" for s in sents), encoding="utf8")
    return ptb_dir


def toy_splits(seed=0, n_train=300, n_valid=80, n_words=30):
    """A train and a valid list of sentences; valid keeps only sentences whose words all occur in train."""
    sents = toy_corpus(seed, n_train + 3 * n_valid, n_words)
    train = sents[:n_train]
    seen = {w for s in train for w in s}
    valid = [s for s in sents[n_train:] if all(w in seen for w in s)][:n_valid]
    return train, valid


def use_src(tmp_root, train=None, valid=None):
    """Import ptb and runlog from src/, point them at tmp_root, and return (ptb, runlog)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import ptb
    import runlog
    if train is not None:
        ptb.PTB = str(write_ptb(tmp_root, train, valid))
    runlog.ROOT = str(tmp_root)
    return ptb, runlog


def load_module_copy(path, name_hint):
    """Import the file at `path` as a fresh module whose folder also holds the current src/runlog.py and src/ptb.py
    (copied), so that a saved copy of a script (for example the version before a fix) runs like the published one.
    Returns (module, folder)."""
    folder = Path(os.environ.get("TMPDIR", "/tmp")) / f"clm_mod_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    for dep in ("runlog.py", "ptb.py", "kn.py", "pair_baselines.py"):
        shutil.copy(SRC / dep, folder / dep)
    target = folder / f"{name_hint}.py"
    shutil.copy(path, target)
    name = f"{name_hint}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, target)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    spec.loader.exec_module(mod)
    return mod, folder


class Tripwire:
    """Context manager: records the paths opened through builtins.open (and np.load) that contain `part`."""

    def __init__(self, part):
        self.part = part
        self.hits = []

    def __enter__(self):
        self._open, self._npload = builtins.open, np.load

        def spy_open(file, *a, **k):
            if self.part in str(file):
                self.hits.append(str(file))
            return self._open(file, *a, **k)

        def spy_npload(file, *a, **k):
            if self.part in str(file):
                self.hits.append(str(file))
            return self._npload(file, *a, **k)

        builtins.open, np.load = spy_open, spy_npload
        return self

    def __exit__(self, *exc):
        builtins.open, np.load = self._open, self._npload
        return False


class Checks:
    """Collects named checks and prints ok / FAIL lines; .done() prints PASS or the count and exits."""

    def __init__(self):
        self.fails = []

    def __call__(self, cond, what):
        print(("ok   " if cond else "FAIL ") + what, flush=True)
        if not cond:
            self.fails.append(what)
        return cond

    def done(self):
        print(f"{len(self.fails)} failed" if self.fails else "PASS", flush=True)
        sys.exit(1 if self.fails else 0)
