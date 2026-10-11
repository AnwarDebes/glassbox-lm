"""Fitting protocol of the CVTM (docs/11 section 11.6; docs/12 F20 b; docs/13 amendments A3 and A4).

Document residues. The documents of the frozen training split are numbered by their order in the
train store (`data/<corpus>/bpe<V>/train/doc_start.npy`; document i covers
doc_start[i] <= p < doc_start[i + 1]). This index is the STABLE document ID of every protocol
here (H9.9 deletes and inserts documents by it). The residue of document i is

    residue(i) = mix64(i XOR 0xC0FFEE) mod 100

with `mix64` the SplitMix64 step of docs/11 section 3 (`tmlm.engine.rng.mix64`: the golden-gamma
increment 0x9E3779B97F4A7C15, then the standard finaliser; docs/12 F20 b).

Held-out training documents:
- H  = residue 0 (docs/11 11.6): the fit documents of the core's MAP-EM (304 documents,
  1,397,702 tokens of the V 8000 train split; `check_h` asserts both and the sha256 of the index
  list that the exploratory run logged).
- H' = residues {0, 1, 2} (docs/13 A4): 0 fits the core's EM ("core_em"), 1 trains the T2 clauses
  ("t2_train"), 2 is used for selection, temperature and mixture weights ("selection"). The T2
  inputs use a KN-5 fitted WITHOUT H'.

Byte accounting (protocol CP, `experiments/cp_causal_perplexity.py`): bits per byte = total
negative log2-likelihood of the subword stream / UTF-8 bytes of the split's text
(`meta.json["n_bytes_utf8"]`, equal to the raw store's `n_bytes`); per-document bytes come from the
raw store's `doc_bytes.npy` (cumulative UTF-8 offsets, one document per token document).

Dataset hashes (as `results/20260923T042350Z_CP_anchor/dataset_hashes.json`): the sha256 entries of
the stores' meta.json files, `<split>_tok_V<V>` and `<split>_text_utf8`, plus `<split>_doc_start_V<V>`
(the document boundaries define the stable IDs and the residues) and `<split>_doc_bytes` (the
per-document byte counts of the bootstrap).

The TEST split is only loaded through `load_eval_split("test", purpose=...)`, which goes through the
counted path of `tmlm.data.corpus_store` (docs/06); nothing else in this module touches it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..engine.rng import mix64

CORPUS = "wikitext103"
VOCAB = 8000
SALT = 0xC0FFEE
MODULUS = 100
H_RESIDUES = (0,)
HPRIME_RESIDUES = (0, 1, 2)
ROLES = {0: "core_em", 1: "t2_train", 2: "selection"}
ROLE_RESIDUE = {v: k for k, v in ROLES.items()}
# docs/12 F20 b and results/20260924T054225Z_cvtm_early_fulldata_exploratory/ (H.n_docs, H.n_tokens,
# H.doc_index_sha256: sha256 of the int64 little-endian array of the H document indices)
EXPECTED_H = {"n_docs": 304, "n_tokens": 1_397_702,
              "doc_index_sha256": "e76e01bff6610a891b119a969d7a6fa094772cf83e4ac7fc1e3be2badb2a4f6e"}
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)


# ------------------------------------------------------------------ residues
def residue(doc_index: int) -> int:
    """mix64(doc_index XOR 0xC0FFEE) mod 100 for one document (scalar reference)."""
    i = int(doc_index)
    if i < 0:
        raise ValueError("document indices are non-negative")
    return mix64(i ^ SALT) % MODULUS


def residues(doc_index: np.ndarray | Sequence[int] | int) -> np.ndarray:
    """Vectorised `residue` (int64 array); equal to the scalar form element by element (tested)."""
    if np.isscalar(doc_index):
        idx = np.arange(int(doc_index), dtype=np.uint64)
    else:
        a = np.asarray(doc_index)
        if a.size and int(a.min()) < 0:
            raise ValueError("document indices are non-negative")
        idx = a.astype(np.uint64)
    z = idx ^ np.uint64(SALT)
    with np.errstate(over="ignore"):
        z = z + _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(MODULUS)).astype(np.int64)


def doc_mask(n_docs: int, keep: Iterable[int]) -> np.ndarray:
    """bool [n_docs]: documents whose residue is in `keep`."""
    keep = tuple(int(r) for r in keep)
    if any(not 0 <= r < MODULUS for r in keep):
        raise ValueError(f"residues lie in 0..{MODULUS - 1}")
    return np.isin(residues(int(n_docs)), np.asarray(keep, dtype=np.int64))


def index_sha256(doc_index: np.ndarray) -> str:
    """sha256 of the int64 little-endian bytes of a document index list (the exploratory run's H hash)."""
    return hashlib.sha256(np.asarray(doc_index).astype("<i8").tobytes()).hexdigest()


# ------------------------------------------------------------------ document selection
@dataclass(frozen=True)
class DocSubset:
    """Whole documents of a stream: tok [n], doc_start [d + 1] (0-based, ends at n), doc_index [d]
    (stable IDs in the source split), residue [d]."""

    tok: np.ndarray
    doc_start: np.ndarray
    doc_index: np.ndarray
    residue: np.ndarray

    @property
    def n_docs(self) -> int:
        return int(self.doc_index.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.tok.shape[0])

    def doc_of(self) -> np.ndarray:
        """int64 [n]: local document number of every position."""
        return np.repeat(np.arange(self.n_docs, dtype=np.int64), np.diff(self.doc_start))

    def role_of(self) -> np.ndarray:
        """int8 [n]: residue of every position's document (0, 1, 2 for the H' roles)."""
        return np.repeat(self.residue.astype(np.int8), np.diff(self.doc_start))


def select_docs(tok: np.ndarray, doc_start: np.ndarray, keep_doc: np.ndarray,
                doc_index: np.ndarray | None = None) -> DocSubset:
    """The documents with keep_doc True, in their original order, as one contiguous stream.
    doc_index: stable IDs of the input documents (default 0..d-1)."""
    ds = np.asarray(doc_start).astype(np.int64)
    keep_doc = np.asarray(keep_doc).astype(bool)
    if keep_doc.shape[0] != ds.shape[0] - 1:
        raise ValueError("keep_doc needs one entry per document")
    ids = np.arange(ds.shape[0] - 1, dtype=np.int64) if doc_index is None else np.asarray(doc_index).astype(np.int64)
    lengths = np.diff(ds)
    keep_tok = np.repeat(keep_doc, lengths)
    sub = np.ascontiguousarray(np.asarray(tok)[keep_tok])
    dsub = np.zeros(int(keep_doc.sum()) + 1, dtype=np.int64)
    dsub[1:] = np.cumsum(lengths[keep_doc])
    sel = ids[keep_doc]
    return DocSubset(tok=sub, doc_start=dsub, doc_index=sel, residue=residues(sel))


def split_by_residue(tok: np.ndarray, doc_start: np.ndarray, held: Iterable[int],
                     doc_index: np.ndarray | None = None) -> tuple[DocSubset, DocSubset]:
    """(held-out documents with residue in `held`, the remaining documents)."""
    ds = np.asarray(doc_start).astype(np.int64)
    ids = np.arange(ds.shape[0] - 1, dtype=np.int64) if doc_index is None else np.asarray(doc_index).astype(np.int64)
    held = tuple(int(r) for r in held)
    m = np.isin(residues(ids), np.asarray(held, dtype=np.int64))
    return select_docs(tok, ds, m, ids), select_docs(tok, ds, ~m, ids)


def prefix_documents(tok: np.ndarray, doc_start: np.ndarray, n_tokens: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The first `n_tokens` tokens of a split cut at a document boundary: the leading whole documents
    whose end is at most n_tokens (at least one document); None = the whole split. The kept
    documents keep their indices 0..d-1, so stable IDs and residues are those of the full split.
    Same cut as `experiments/m3_tiny_lm.py` slice_train for every prefix that is not a boundary itself."""
    ds = np.asarray(doc_start).astype(np.int64)
    if n_tokens is None or int(n_tokens) >= int(ds[-1]):
        return np.asarray(tok), ds
    d = int(np.searchsorted(ds, int(n_tokens), side="right")) - 1
    d = max(1, d)
    end = int(ds[d])
    return np.asarray(tok)[:end], ds[:d + 1].copy()


def check_h(n_docs_h: int, n_tokens_h: int, doc_index_h: np.ndarray) -> None:
    """Assert the H selection of docs/12 F20 b on the full V 8000 train split."""
    got = {"n_docs": int(n_docs_h), "n_tokens": int(n_tokens_h), "doc_index_sha256": index_sha256(doc_index_h)}
    if got != EXPECTED_H:
        raise AssertionError(f"H selection differs from docs/12 F20 b: expected {EXPECTED_H}, got {got}")


# ------------------------------------------------------------------ data loading
@dataclass(frozen=True)
class EvalSplit:
    """An evaluation split with its byte accounting."""

    split: str
    tok: np.ndarray            # uint16 [n]
    doc_start: np.ndarray      # int64 [d + 1]
    doc_bytes: np.ndarray      # int64 [d]: UTF-8 bytes per document
    n_bytes: int               # UTF-8 bytes of the split (the denominator of bits per byte)
    n_words: int
    meta: dict
    raw_meta: dict

    @property
    def n_docs(self) -> int:
        return int(self.doc_start.shape[0] - 1)

    def doc_of(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_docs, dtype=np.int64), np.diff(self.doc_start))


def _metas(corpus: str, vocab: int, split: str) -> tuple[dict, dict]:
    from ..data.corpus_store import raw_dir, split_dir

    meta = json.loads((split_dir(corpus, vocab, split) / "meta.json").read_text())
    raw = json.loads((raw_dir(corpus, split) / "meta.json").read_text())
    return meta, raw


def load_train(corpus: str = CORPUS, vocab: int = VOCAB, check: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(tok, doc_start int64) of the train split (tok memory-mapped; check re-hashes every array)."""
    from ..data.corpus_store import load_split

    tr = load_split(corpus, vocab, "train", check=check)
    return np.asarray(tr.tok), np.asarray(tr.doc_start).astype(np.int64)


def load_eval_split(split: str, corpus: str = CORPUS, vocab: int = VOCAB, purpose: str | None = None,
                    check: bool = True) -> EvalSplit:
    """Validation (or, with a purpose, the counted test access) with its byte accounting."""
    if split not in ("validation", "test"):
        raise ValueError("evaluation splits are 'validation' and 'test'")
    if split == "test" and not purpose:
        raise ValueError("the test split needs a purpose (the access is counted, docs/06)")
    from ..data.corpus_store import load_split, raw_dir

    sp = load_split(corpus, vocab, split, purpose=purpose, check=check)
    meta, raw = _metas(corpus, vocab, split)
    doc_bytes_off = np.load(raw_dir(corpus, split) / "doc_bytes.npy").astype(np.int64)
    ds = np.asarray(sp.doc_start).astype(np.int64)
    n_bytes = int(meta["n_bytes_utf8"])
    if int(raw["n_bytes"]) != n_bytes or int(doc_bytes_off[-1]) != n_bytes or doc_bytes_off.shape[0] != ds.shape[0]:
        raise RuntimeError(f"{split}: byte offsets do not match the token documents")
    return EvalSplit(split=split, tok=np.ascontiguousarray(np.asarray(sp.tok)), doc_start=ds,
                     doc_bytes=np.diff(doc_bytes_off), n_bytes=n_bytes, n_words=int(meta["n_words"]),
                     meta=meta, raw_meta=raw)


def dataset_hashes(splits: Sequence[str] = ("train", "validation"), corpus: str = CORPUS,
                   vocab: int = VOCAB) -> dict[str, str]:
    """The sha256 digests of the stores (from their meta.json files) for `runlog.write_dataset_hashes`.
    Only the digests are read: asking for "test" here does not load the test text or tokens."""
    out: dict[str, str] = {}
    for split in splits:
        meta, raw = _metas(corpus, vocab, split)
        out[f"{split}_tok_V{vocab}"] = meta["sha256"]["tok.npy"]
        out[f"{split}_doc_start_V{vocab}"] = meta["sha256"]["doc_start.npy"]
        if split != "train":
            out[f"{split}_text_utf8"] = raw["sha256"]["text.utf8"]
            out[f"{split}_doc_bytes"] = raw["sha256"]["doc_bytes.npy"]
    return out


def write_dataset_hashes(splits: Sequence[str] = ("train", "validation"), corpus: str = CORPUS,
                         vocab: int = VOCAB) -> dict[str, str]:
    """`dataset_hashes` written into the current run (no-op outside run_logged.py); returns them."""
    from ..common import runlog

    h = dataset_hashes(splits, corpus, vocab)
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes(h, on_conflict="replace")
    return h


# ------------------------------------------------------------------ document-block bootstrap
N_BOOT = 2000
BOOT_SEED = (20260924, 1106, 1)          # the seed of the exploratory CVTM runs (comparable resamples)


def bootstrap_index(n_docs: int, n_boot: int = N_BOOT, seed: Sequence[int] = BOOT_SEED) -> np.ndarray:
    """int64 [n_boot, n_docs]: documents drawn with replacement (numpy SeedSequence(seed))."""
    rng = np.random.default_rng(np.random.SeedSequence([int(s) for s in seed]))
    return rng.integers(0, int(n_docs), size=(int(n_boot), int(n_docs)))


def bootstrap_bpb(doc_bits: np.ndarray, doc_bytes: np.ndarray, idx: np.ndarray,
                  doc_bits_ref: np.ndarray | None = None, const: float = 0.0) -> dict:
    """Point estimate and 95 percent percentile interval of (sum bits - sum ref bits) / sum bytes over
    resampled documents (the ratio is formed per resample, docs/13 A3), minus `const`. Labelled derived."""
    d = np.asarray(doc_bits, dtype=np.float64) - (0.0 if doc_bits_ref is None else np.asarray(doc_bits_ref, dtype=np.float64))
    by = np.asarray(doc_bytes, dtype=np.float64)
    point = float(d.sum() / by.sum()) - const
    stats = d[idx].sum(axis=1) / by[idx].sum(axis=1) - const
    return {"point": point, "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "n_boot": int(idx.shape[0]), "n_docs": int(idx.shape[1])}


# ------------------------------------------------------------------ summaries
def selection_summary(n_docs: int, doc_start: np.ndarray, keep: Iterable[int]) -> dict:
    """Documents, tokens and the index sha256 per residue of `keep`, and in total."""
    ds = np.asarray(doc_start).astype(np.int64)
    r = residues(int(n_docs))
    lengths = np.diff(ds)
    out: dict = {"per_residue": {}}
    allm = np.zeros(int(n_docs), dtype=bool)
    for k in keep:
        m = r == int(k)
        allm |= m
        out["per_residue"][str(int(k))] = {"role": ROLES.get(int(k)), "n_docs": int(m.sum()),
                                           "n_tokens": int(lengths[m].sum()),
                                           "doc_index_sha256": index_sha256(np.flatnonzero(m))}
    out["n_docs"] = int(allm.sum())
    out["n_tokens"] = int(lengths[allm].sum())
    out["doc_index_sha256"] = index_sha256(np.flatnonzero(allm))
    return out
