"""Token corpus store (docs/11 section 1.1; `tmlm.engine.api.CorpusArrays`) and raw article store.

Layout under the data root (`$TMLM_DATA_DIR`, default `<project>/data`):

    <corpus>/raw/<split>/text.utf8        the exact UTF-8 text that is tokenized, all articles
                                           back to back (bits per byte counts these bytes)
    <corpus>/raw/<split>/doc_bytes.npy    u64 [N_doc + 1] byte offsets of the articles in text.utf8
    <corpus>/raw/<split>/titles.json      the article titles, in order
    <corpus>/raw/<split>/meta.json        row counts, byte counts, sha256 of every file above
    <corpus>/bpe<V>/<split>/tok.npy       u16 [N_tok]      token ids (CorpusArrays.tok)
    <corpus>/bpe<V>/<split>/doc_start.npy u32 [N_doc + 1]  one document per article
    <corpus>/bpe<V>/<split>/word_id.npy   u32 [N_tok]      space-delimited word of every token,
                                           NO_WORD for whitespace-only tokens (whole-word eval)
    <corpus>/bpe<V>/<split>/kind_map.npy  u16 [4, V]       identity plus the 3 cluster levels
    <corpus>/bpe<V>/<split>/stop.npy      u8  [V]          N_STOP most frequent TRAIN tokens
    <corpus>/bpe<V>/<split>/sym_freq.npy  u32 [N_SYM]      TRAIN frequency of every symbol
    <corpus>/bpe<V>/<split>/targets.npy   u32 [N_ex]       all positions
    <corpus>/bpe<V>/<split>/meta.json     shapes, byte count, sha256 of every array

kind_map, stop and sym_freq are computed from the TRAIN split only and copied into every
split directory so that a split directory is a complete CorpusArrays.

Files are only ever CREATED here (protocol rule 9): `write_array` refuses to overwrite,
and a builder that finds an existing file re-reads it and checks its recorded sha256.

WikiText-103 raw rows are lines. An empty row is an empty line of the original file; a
non-empty row ends with "\\n". `rows_to_articles` rebuilds the original file text
(empty row -> "\\n") and cuts it at top-level headings " = Title = " (exactly one "=" on
each side, an empty row before and after). Rows before the first heading are attached to the
first article.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..common import hashing
from ..common.runlog import project_root
from ..engine.api import CorpusArrays

N_STOP = 64
NO_WORD = 0xFFFFFFFF
STORE_ARRAYS = ("tok", "doc_start", "word_id", "kind_map", "stop", "sym_freq", "targets")
STORE_DTYPES = {
    "tok": np.uint16, "doc_start": np.uint32, "word_id": np.uint32, "kind_map": np.uint16,
    "stop": np.uint8, "sym_freq": np.uint32, "targets": np.uint32,
}


def data_root() -> Path:
    """`$TMLM_DATA_DIR` when set, else `<project>/data`."""
    override = os.environ.get("TMLM_DATA_DIR")
    if override:
        return Path(override)
    return project_root() / "data"


# ------------------------------------------------------------------ create-only file helpers
def write_new_bytes(path: Path, data: bytes) -> str:
    """Create `path` with `data` (never overwrite) and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)
    return hashing.sha256_bytes(data)


def write_new_json(path: Path, obj: object) -> str:
    text = hashing.canonical_json(obj) + "\n"
    return write_new_bytes(path, text.encode("utf-8"))


def write_array(path: Path, arr: np.ndarray) -> str:
    """Create `path` (.npy, never overwrite) and return `hashing.sha256_array(arr)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.save(handle, np.ascontiguousarray(arr))
    return hashing.sha256_array(arr)


def load_array(path: Path, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


# ------------------------------------------------------------------ articles
def is_top_heading(line: str) -> bool:
    """A line of the form ' = Title = \\n' with exactly one '=' on each side. Not sufficient on its
    own: the Wikipedia extraction also left equation fragments such as ' = 1 rad / s and a
    nominal impedance k = \\n' as whole rows. A real heading is surrounded by empty rows
    (`rows_to_articles` checks that); the two rules together give the official article counts
    for validation and test (60 and 60) and 28472 for train (the dataset page says 28,475: three
    train headings were broken over two rows by the extraction, for example row 8030
    ' = De dezas a espwa : interviews with persons affected by the 2010 Hai', so those three
    articles are merged into their predecessor; measured 2026-09-22)."""
    return (line.startswith(" = ") and not line.startswith(" = = ")
            and line.endswith(" = \n") and not line.endswith(" = = \n"))


def rows_to_articles(rows: Iterable[str]) -> tuple[list[str], list[str]]:
    """Cut the rows of one split into (texts, titles). An article starts at a top-level heading
    row that has an empty row before it (or is the first row) and an empty row after it. Every
    byte of the original file is in exactly one article text (empty row -> "\\n"); rows before
    the first heading go to the first article."""
    rows = list(rows)
    n = len(rows)
    texts: list[str] = []
    titles: list[str] = []
    current: list[str] = []
    for i, row in enumerate(rows):
        if (is_top_heading(row) and (i == 0 or rows[i - 1] == "")
                and i + 1 < n and rows[i + 1] == ""):
            if titles:
                texts.append("".join(current))
                current = []
            # rows before the first heading stay in `current`: first article
            titles.append(row.strip()[1:-1].strip())
        current.append(row if row else "\n")
    if titles:
        texts.append("".join(current))
    elif current:
        titles.append("")
        texts.append("".join(current))
    return texts, titles


@dataclass(frozen=True)
class RawSplit:
    """The raw article store of one split."""

    text: bytes
    doc_bytes: np.ndarray  # u64 [N_doc + 1]
    titles: list[str]

    @property
    def n_doc(self) -> int:
        return int(self.doc_bytes.shape[0] - 1)

    def article(self, d: int) -> str:
        return self.text[int(self.doc_bytes[d]):int(self.doc_bytes[d + 1])].decode("utf-8")

    def articles(self) -> list[str]:
        return [self.article(d) for d in range(self.n_doc)]


def raw_dir(corpus: str, split: str, root: Path | None = None) -> Path:
    return (root or data_root()) / corpus / "raw" / split


def write_raw_split(corpus: str, split: str, texts: Sequence[str], titles: Sequence[str],
                    source: dict, root: Path | None = None) -> dict:
    """Create the raw store of one split and return its meta (with the sha256 of every file)."""
    out = raw_dir(corpus, split, root)
    encoded = [t.encode("utf-8") for t in texts]
    doc_bytes = np.zeros(len(encoded) + 1, dtype=np.uint64)
    doc_bytes[1:] = np.cumsum([len(b) for b in encoded], dtype=np.uint64)
    blob = b"".join(encoded)
    meta = {
        "corpus": corpus, "split": split, "n_doc": len(encoded), "n_bytes": len(blob),
        "n_chars": sum(len(t) for t in texts), "source": source,
        "sha256": {
            "text.utf8": write_new_bytes(out / "text.utf8", blob),
            "doc_bytes.npy": write_array(out / "doc_bytes.npy", doc_bytes),
            "titles.json": write_new_json(out / "titles.json", list(titles)),
        },
    }
    write_new_json(out / "meta.json", meta)
    return meta


def load_raw_split(corpus: str, split: str, root: Path | None = None, purpose: str | None = None,
                   check: bool = True) -> RawSplit:
    """Load a raw split. A test split needs a `purpose`: the access is counted (docs/06)."""
    if split == "test":
        from .evalsets import record_test_access
        if not purpose:
            raise ValueError("loading a test split needs a purpose (the access is counted)")
        record_test_access("corpus_store.load_raw_split", purpose, root=root)
    d = raw_dir(corpus, split, root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    text = (d / "text.utf8").read_bytes()
    doc_bytes = np.load(d / "doc_bytes.npy")
    titles = json.loads((d / "titles.json").read_text(encoding="utf-8"))
    if check:
        if hashing.sha256_bytes(text) != meta["sha256"]["text.utf8"]:
            raise ValueError(f"{d}/text.utf8: sha256 mismatch with meta.json")
        if hashing.sha256_array(doc_bytes) != meta["sha256"]["doc_bytes.npy"]:
            raise ValueError(f"{d}/doc_bytes.npy: sha256 mismatch with meta.json")
    return RawSplit(text=text, doc_bytes=doc_bytes, titles=titles)


# ------------------------------------------------------------------ token store
def split_dir(corpus: str, vocab: int, split: str, root: Path | None = None) -> Path:
    return (root or data_root()) / corpus / f"bpe{vocab}" / split


def write_split_arrays(corpus: str, vocab: int, split: str, arrays: dict[str, np.ndarray],
                       root: Path | None = None) -> dict[str, str]:
    """Create the .npy files named in `arrays` (a subset of STORE_ARRAYS) and return their hashes.
    The dtype of every array is enforced (docs/11 section 1.1)."""
    out = split_dir(corpus, vocab, split, root)
    hashes = {}
    for name, arr in arrays.items():
        if name not in STORE_ARRAYS:
            raise KeyError(name)
        if arr.dtype != STORE_DTYPES[name]:
            raise TypeError(f"{name}: expected {np.dtype(STORE_DTYPES[name])}, got {arr.dtype}")
        hashes[f"{name}.npy"] = write_array(out / f"{name}.npy", arr)
    return hashes


def write_split_meta(corpus: str, vocab: int, split: str, meta: dict, root: Path | None = None) -> str:
    return write_new_json(split_dir(corpus, vocab, split, root) / "meta.json", meta)


def load_split(corpus: str, vocab: int, split: str, root: Path | None = None,
               purpose: str | None = None, mmap: bool = True, check: bool = False) -> CorpusArrays:
    """CorpusArrays of one split. A test split needs a `purpose` (the access is counted).
    `check=True` re-hashes every array against meta.json."""
    if split == "test":
        from .evalsets import record_test_access
        if not purpose:
            raise ValueError("loading a test split needs a purpose (the access is counted)")
        record_test_access("corpus_store.load_split", purpose, root=root)
    d = split_dir(corpus, vocab, split, root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    arrays = {name: load_array(d / f"{name}.npy", mmap=mmap) for name in STORE_ARRAYS if name != "word_id"}
    if check:
        for name, arr in arrays.items():
            if hashing.sha256_array(np.asarray(arr)) != meta["sha256"][f"{name}.npy"]:
                raise ValueError(f"{d}/{name}.npy: sha256 mismatch with meta.json")
    return CorpusArrays(**arrays)


def load_word_id(corpus: str, vocab: int, split: str, root: Path | None = None) -> np.ndarray:
    """word_id of a split; loading it alone counts as no test access (it holds no text)."""
    return load_array(split_dir(corpus, vocab, split, root) / "word_id.npy")
