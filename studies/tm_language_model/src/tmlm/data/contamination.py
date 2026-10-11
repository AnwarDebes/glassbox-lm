"""Test/train and validation/train n-gram overlap (docs/06 "Contamination rules"; n = 8 and 13
at the BPE-8000 token level).

Every n-gram lies inside one document. An n-gram of an evaluation split is SEEN when the same
token sequence occurs anywhere in the train split. Matching is by a 64-bit polynomial hash of
the n token ids (numpy uint64 arithmetic, wrapping), sorted with np.unique and looked up with
np.searchsorted (no Python dicts). A false
match needs a hash collision between one evaluation n-gram and one of the N_train distinct
train n-grams: probability about N_train / 2^64 per evaluation n-gram, below 1e-10 here, so the
reported fractions are exact up to that.

Reported per evaluation split and n:
- ngram_seen_fraction: seen n-grams / all n-grams of the split;
- token_covered_fraction: tokens lying inside at least one seen n-gram / all tokens;
- per document: doc_frac = seen n-grams / n-grams of the document (0 for a document shorter
  than n), and doc_keep = 1 iff doc_frac < DEOVERLAP_THRESHOLD = 0.5. The de-overlapped subset
  of docs/06 is the set of documents with doc_keep = 1 at n = 13. The rule is stated in the
  meta.json of every file so any other threshold can be applied from doc_frac later.
Files: <corpus>/contamination_bpe<V>/<split>_<n>gram.npz with `seen` (u8 per n-gram start
position, in position order), `start` (u32 start positions), `token_covered` (u8 [N_tok]),
`doc_frac` (f32 [N_doc]), `doc_keep` (u8 [N_doc]); plus .meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import hashing
from .corpus_store import data_root, write_new_json
from .evalsets import combined_hash, record_test_access

NGRAM_ORDERS = (8, 13)
DEOVERLAP_THRESHOLD = 0.5
_P = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier; arithmetic wraps mod 2^64


def ngram_starts(doc_start: np.ndarray, n: int) -> np.ndarray:
    """u32 start positions i of every n-gram lying inside one document (i + n <= doc end)."""
    doc_start = np.asarray(doc_start, dtype=np.int64)
    lens = np.diff(doc_start)
    starts = []
    for d in np.flatnonzero(lens >= n):
        starts.append(np.arange(doc_start[d], doc_start[d + 1] - n + 1, dtype=np.int64))
    if not starts:
        return np.zeros(0, dtype=np.uint32)
    return np.concatenate(starts).astype(np.uint32)


def ngram_hashes(tok: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """u64 polynomial hash of tok[i : i + n] for every start i."""
    tok = np.asarray(tok)
    idx = np.asarray(starts, dtype=np.int64)
    h = np.zeros(idx.shape[0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _P + tok[idx + k].astype(np.uint64) + np.uint64(1)
    return h


def train_ngram_set(train_tok: np.ndarray, train_doc_start: np.ndarray, n: int) -> np.ndarray:
    """Sorted unique hashes of every train n-gram."""
    starts = ngram_starts(train_doc_start, n)
    return np.unique(ngram_hashes(train_tok, starts, n))


def seen_mask(train_set: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    if train_set.shape[0] == 0 or hashes.shape[0] == 0:
        return np.zeros(hashes.shape[0], dtype=bool)
    idx = np.searchsorted(train_set, hashes)
    idx = np.minimum(idx, train_set.shape[0] - 1)
    return train_set[idx] == hashes


def overlap(train_set: np.ndarray, tok: np.ndarray, doc_start: np.ndarray, n: int,
            threshold: float = DEOVERLAP_THRESHOLD) -> dict:
    """The arrays and summary numbers described in the module docstring, for one split."""
    n_tok = int(np.asarray(tok).shape[0])
    doc_start = np.asarray(doc_start, dtype=np.int64)
    n_doc = doc_start.shape[0] - 1
    starts = ngram_starts(doc_start, n)
    seen = seen_mask(train_set, ngram_hashes(tok, starts, n))
    covered = np.zeros(n_tok + 1, dtype=np.int64)
    s = starts[seen].astype(np.int64)
    np.add.at(covered, s, 1)
    np.add.at(covered, s + n, -1)
    token_covered = (np.cumsum(covered[:n_tok]) > 0).astype(np.uint8)
    doc_of_start = np.searchsorted(doc_start, starts.astype(np.int64), side="right") - 1
    per_doc_total = np.bincount(doc_of_start, minlength=n_doc)
    per_doc_seen = np.bincount(doc_of_start, weights=seen.astype(np.float64), minlength=n_doc)
    with np.errstate(divide="ignore", invalid="ignore"):
        doc_frac = np.where(per_doc_total > 0, per_doc_seen / np.maximum(per_doc_total, 1), 0.0).astype(np.float32)
    doc_keep = (doc_frac < threshold).astype(np.uint8)
    return {
        "seen": seen.astype(np.uint8), "start": starts, "token_covered": token_covered,
        "doc_frac": doc_frac, "doc_keep": doc_keep,
        "n": n, "n_ngrams": int(starts.shape[0]), "n_seen": int(seen.sum()),
        "ngram_seen_fraction": float(seen.mean()) if starts.shape[0] else 0.0,
        "token_covered_fraction": float(token_covered.mean()) if n_tok else 0.0,
        "n_doc": int(n_doc), "n_doc_keep": int(doc_keep.sum()),
        "tokens_in_kept_docs": int(np.diff(doc_start)[doc_keep.astype(bool)].sum()),
        "deoverlap_rule": f"doc_keep = 1 iff (seen {n}-grams / {n}-grams of the document) < {threshold}",
    }


def contamination_dir(corpus: str, vocab: int, root: Path | None = None) -> Path:
    return (root or data_root()) / corpus / f"contamination_bpe{vocab}"


def save_overlap(corpus: str, vocab: int, split: str, built: dict, root: Path | None = None) -> dict:
    n = built["n"]
    path = contamination_dir(corpus, vocab, root) / f"{split}_{n}gram.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in built.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in built.items() if not isinstance(v, np.ndarray)}
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    hashes = {k: hashing.sha256_array(v) for k, v in arrays.items()}
    meta = {"corpus": corpus, "vocab": vocab, "split": split, **scalars, "sha256": hashes,
            "sha256_combined": combined_hash(hashes), "sha256_file": hashing.sha256_file(path)}
    write_new_json(path.with_suffix(".meta.json"), meta)
    return meta


def load_overlap(corpus: str, vocab: int, split: str, n: int, root: Path | None = None,
                 purpose: str | None = None) -> dict:
    if split == "test":
        if not purpose:
            raise ValueError("loading test overlap flags needs a purpose (the access is counted)")
        record_test_access("contamination.load_overlap", purpose, root=root)
    path = contamination_dir(corpus, vocab, root) / f"{split}_{n}gram.npz"
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return {**arrays, "meta": meta}
