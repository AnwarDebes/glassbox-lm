"""Byte-level BPE (Hugging Face `tokenizers`), trained on the TRAIN split only (docs/06, docs/13
section 4). Vocabulary sizes 2000, 8000, 16000, 32000; every id fits u16 (V <= 32768).

Tokenizer directory: data/tokenizers/bpe_<V>/tokenizer.json plus meta.json with its sha256 and
the training configuration. Training is deterministic (checked on a slice in
tests/test_data_store.py: two trainings give byte-identical JSON) and uses at most the 6
threads of the CPU quota (RAYON_NUM_THREADS from env/env.sh). env/env.sh exports
TOKENIZERS_PARALLELISM=false; `train_bpe` and `encode_docs` enable it for their own calls
(the variable is read by the library at each call) and restore it afterwards.

Word ids for the whole-word evaluation variant (docs/06): a "word" is a maximal run of
non-whitespace characters of the raw text. The ByteLevel pre-tokenizer never lets a token
cross a whitespace boundary and a token is either whitespace-only, or a leading space plus
content, or content only, so a token starts a new word iff it has content and (it starts
with whitespace, or the previous token was whitespace-only, or it is the first token of its
document). `word_ids` applies that rule with vectorized numpy; tests check it against the
character offsets of the tokenizer.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..common import hashing
from .corpus_store import NO_WORD, data_root, write_new_bytes, write_new_json

MIN_FREQUENCY = 2


@contextmanager
def _parallel_tokenizers() -> Iterator[None]:
    old = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = old


def new_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(texts: Iterable[str], vocab_size: int, min_frequency: int = MIN_FREQUENCY):
    """Train a byte-level BPE tokenizer of `vocab_size` symbols on `texts` (train split only)."""
    from tokenizers import pre_tokenizers, trainers

    if vocab_size > 32768:
        raise ValueError("token ids must fit u16: vocab_size <= 32768")
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    with _parallel_tokenizers():
        tok.train_from_iterator(texts, trainer=trainer)
    return tok


def tokenizer_dir(vocab_size: int, root: Path | None = None) -> Path:
    return (root or data_root()) / "tokenizers" / f"bpe_{vocab_size}"


def save_tokenizer(tok, vocab_size: int, trained_on: dict, root: Path | None = None) -> dict:
    """Create tokenizer.json and meta.json (never overwrite); return the meta."""
    d = tokenizer_dir(vocab_size, root)
    blob = tok.to_str().encode("utf-8")
    meta = {
        "vocab_size_requested": vocab_size, "vocab_size": tok.get_vocab_size(),
        "model": "byte-level BPE (tokenizers)", "min_frequency": MIN_FREQUENCY,
        "trained_on": trained_on, "sha256": {"tokenizer.json": write_new_bytes(d / "tokenizer.json", blob)},
    }
    write_new_json(d / "meta.json", meta)
    return meta


def load_tokenizer(vocab_size: int, root: Path | None = None, check: bool = True):
    from tokenizers import Tokenizer

    d = tokenizer_dir(vocab_size, root)
    blob = (d / "tokenizer.json").read_bytes()
    if check:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if hashing.sha256_bytes(blob) != meta["sha256"]["tokenizer.json"]:
            raise ValueError(f"{d}/tokenizer.json: sha256 mismatch with meta.json")
    return Tokenizer.from_str(blob.decode("utf-8"))


def tokenizer_sha256(vocab_size: int, root: Path | None = None) -> str:
    return hashing.sha256_file(tokenizer_dir(vocab_size, root) / "tokenizer.json")


# ------------------------------------------------------------------ byte-level alphabet
def _unicode_to_byte() -> dict[str, int]:
    """Inverse of the GPT-2 bytes_to_unicode map used by the ByteLevel pre-tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok) -> list[bytes]:
    """The byte string of every token id, in id order."""
    inv = _unicode_to_byte()
    vocab = tok.get_vocab()
    out = [b""] * tok.get_vocab_size()
    for text, tid in vocab.items():
        out[tid] = bytes(inv[ch] for ch in text)
    return out


_WS = frozenset(b" \t\n\r\x0b\x0c")


def token_whitespace_tables(tok) -> tuple[np.ndarray, np.ndarray]:
    """(ws_lead, all_ws): bool [V]. ws_lead[t]: the token starts with a whitespace byte.
    all_ws[t]: every byte is whitespace (or the token is empty)."""
    tb = token_bytes(tok)
    ws_lead = np.array([len(b) > 0 and b[0] in _WS for b in tb], dtype=bool)
    all_ws = np.array([all(x in _WS for x in b) for b in tb], dtype=bool)
    return ws_lead, all_ws


# ------------------------------------------------------------------ encoding
def encode_docs(tok, texts: Sequence[str], batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Encode documents to (tok u16 [N_tok], doc_start u32 [N_doc + 1])."""
    ids: list[np.ndarray] = []
    lengths = np.zeros(len(texts), dtype=np.int64)
    with _parallel_tokenizers():
        for lo in range(0, len(texts), batch):
            encs = tok.encode_batch(list(texts[lo:lo + batch]))
            for k, e in enumerate(encs):
                arr = np.asarray(e.ids, dtype=np.uint16)
                ids.append(arr)
                lengths[lo + k] = arr.shape[0]
    doc_start = np.zeros(len(texts) + 1, dtype=np.uint32)
    if len(texts):
        doc_start[1:] = np.cumsum(lengths)
    if doc_start[-1] > 0xFFFFFFFF:
        raise OverflowError("N_tok does not fit u32")
    flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint16)
    if flat.size and int(flat.max()) >= tok.get_vocab_size():
        raise ValueError("token id outside the vocabulary")
    return flat, doc_start


def word_ids(tokens: np.ndarray, doc_start: np.ndarray, ws_lead: np.ndarray, all_ws: np.ndarray) -> np.ndarray:
    """u32 [N_tok]: index of the space-delimited word of every token (counted over the whole
    split, document boundaries are word boundaries); NO_WORD for whitespace-only tokens."""
    n = tokens.shape[0]
    content = ~all_ws[tokens]
    prev_ws = np.ones(n, dtype=bool)
    if n > 1:
        prev_ws[1:] = all_ws[tokens[:-1]]
    first = np.zeros(n, dtype=bool)
    first[doc_start[:-1][doc_start[:-1] < n]] = True
    new_word = content & (ws_lead[tokens] | prev_ws | first)
    wid = np.cumsum(new_word, dtype=np.int64) - 1
    out = np.where(content, wid, NO_WORD).astype(np.uint32)
    if content.any() and wid[content].min() < 0:
        raise AssertionError("a content token before the first word start")
    return out


def word_ids_from_offsets(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    """Reference word ids from character offsets (slow; used by the tests)."""
    import re

    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    starts = np.array([s for s, _ in words], dtype=np.int64)
    out = []
    for s, e in offsets:
        piece = text[s:e]
        k = next((i for i, ch in enumerate(piece) if not ch.isspace()), None)
        if k is None:
            out.append(NO_WORD)
            continue
        w = int(np.searchsorted(starts, s + k, side="right") - 1)
        out.append(w)
    return out
