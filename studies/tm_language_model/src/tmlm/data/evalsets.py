"""Frozen evaluation mask sets (docs/06, docs/13 section 4) and the test-set access counter.

Two variants per (corpus, vocab, split in {validation, test}):
- subword: every token position is hidden independently with probability RATE = 0.15 from a
  fixed seed; targets = the hidden positions. For validation, if more than MAX_VALID_TARGETS
  (200000) positions are hidden, a fixed random subset of that size is the target list; for
  test every hidden position is a target. meta.json states which case applied.
- wholeword: every space-delimited word (corpus_store word_id) is hidden entirely with
  probability RATE; targets = every hidden piece; whitespace-only tokens (NO_WORD) are never
  hidden and never targets. The npz carries word_id so an evaluator can score whole words.

The seed of a set is SeedSequence([BASE_SEED, vocab, split code, variant code]), so every set is
reproducible from this file alone; the arrays are hashed (sha256 of every array in meta.json
and one combined digest).

Test-set access counter (docs/06 "every test-set access is counted"): every loader of a test
split calls `record_test_access(caller, purpose)`, which appends one JSON line
{utc, caller, purpose, pid} to <data root>/test_access_log.jsonl and returns the new count.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..common import hashing
from .corpus_store import NO_WORD, data_root, write_new_json

RATE = 0.15
MAX_VALID_TARGETS = 200_000
BASE_SEED = 20260922
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
VARIANT_CODE = {"subword": 1, "wholeword": 2}
ACCESS_LOG = "test_access_log.jsonl"


# ------------------------------------------------------------------ access counter
def access_log_path(root: Path | None = None) -> Path:
    return (root or data_root()) / ACCESS_LOG


def record_test_access(caller: str, purpose: str, root: Path | None = None) -> int:
    """Append one line to the test access log and return the total number of recorded accesses."""
    if not purpose or not purpose.strip():
        raise ValueError("a test-set access needs a non-empty purpose")
    path = access_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "caller": caller,
                       "purpose": purpose, "pid": os.getpid()}, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return test_access_count(root)


def test_access_count(root: Path | None = None) -> int:
    path = access_log_path(root)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


# ------------------------------------------------------------------ building
def evalset_seed(vocab: int, split: str, variant: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([BASE_SEED, int(vocab), SPLIT_CODE[split], VARIANT_CODE[variant]])


def build_subword(n_tok: int, vocab: int, split: str, rate: float = RATE,
                  max_targets: int | None = None) -> dict:
    rng = np.random.default_rng(evalset_seed(vocab, split, "subword"))
    hidden = (rng.random(n_tok) < rate).astype(np.uint8)
    targets = np.flatnonzero(hidden).astype(np.uint32)
    rule = "all hidden positions"
    if max_targets is not None and targets.shape[0] > max_targets:
        pick = np.sort(rng.choice(targets.shape[0], size=max_targets, replace=False))
        targets = targets[pick]
        rule = f"fixed random subset of {max_targets} hidden positions"
    return {"hidden": hidden, "targets": targets, "rule": rule, "rate": rate}


def build_wholeword(word_id: np.ndarray, vocab: int, split: str, rate: float = RATE) -> dict:
    word_id = np.asarray(word_id, dtype=np.uint32)
    content = word_id != NO_WORD
    n_words = int(word_id[content].max()) + 1 if content.any() else 0
    rng = np.random.default_rng(evalset_seed(vocab, split, "wholeword"))
    hidden_word = rng.random(n_words) < rate
    hidden = np.zeros(word_id.shape[0], dtype=np.uint8)
    hidden[content] = hidden_word[word_id[content]]
    targets = np.flatnonzero(hidden).astype(np.uint32)
    return {"hidden": hidden, "targets": targets, "word_id": word_id,
            "hidden_words": np.flatnonzero(hidden_word).astype(np.uint32),
            "rule": "every piece of every hidden word; whitespace-only tokens are never hidden",
            "rate": rate, "n_words": n_words}


# ------------------------------------------------------------------ storage
def evalset_dir(corpus: str, vocab: int, root: Path | None = None) -> Path:
    return (root or data_root()) / corpus / f"bpe{vocab}" / "evalsets"


def evalset_path(corpus: str, vocab: int, split: str, variant: str, root: Path | None = None) -> Path:
    return evalset_dir(corpus, vocab, root) / f"{split}_{variant}.npz"


def combined_hash(hashes: dict[str, str]) -> str:
    return hashing.sha256_bytes(hashing.canonical_json(dict(sorted(hashes.items()))).encode("utf-8"))


def save_evalset(corpus: str, vocab: int, split: str, variant: str, built: dict,
                 root: Path | None = None) -> dict:
    """Create <split>_<variant>.npz and .meta.json (never overwrite); return the meta."""
    path = evalset_path(corpus, vocab, split, variant, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in built.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in built.items() if not isinstance(v, np.ndarray)}
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    hashes = {k: hashing.sha256_array(v) for k, v in arrays.items()}
    meta = {"corpus": corpus, "vocab": vocab, "split": split, "variant": variant,
            "seed": {"base": BASE_SEED, "vocab": vocab, "split_code": SPLIT_CODE[split],
                     "variant_code": VARIANT_CODE[variant]},
            "n_tok": int(arrays["hidden"].shape[0]), "n_hidden": int(arrays["hidden"].sum()),
            "n_targets": int(arrays["targets"].shape[0]), **scalars,
            "sha256": hashes, "sha256_combined": combined_hash(hashes),
            "sha256_file": hashing.sha256_file(path)}
    write_new_json(path.with_suffix(".meta.json"), meta)
    return meta


def load_evalset(corpus: str, vocab: int, split: str, variant: str, root: Path | None = None,
                 purpose: str | None = None, check: bool = True) -> dict:
    """Arrays of a frozen set plus its meta. A test split needs a `purpose` (counted)."""
    if split == "test":
        if not purpose:
            raise ValueError("loading a test eval set needs a purpose (the access is counted)")
        record_test_access("evalsets.load_evalset", purpose, root=root)
    path = evalset_path(corpus, vocab, split, variant, root)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if check:
        for k, v in arrays.items():
            if hashing.sha256_array(v) != meta["sha256"][k]:
                raise ValueError(f"{path}:{k}: sha256 mismatch with meta")
    return {**arrays, "meta": meta}
