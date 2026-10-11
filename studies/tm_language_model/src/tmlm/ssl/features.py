"""Frozen self-supervised representations of downstream texts (exploratory transfer test, docs/12 F23).

A downstream text becomes ONE document of a new token corpus that the engines can read:

    text -> wikitext_style(text) + " \\n"  (WikiText-103 surface form, one END token appended)
         -> BPE-8k ids of data/tokenizers/bpe_8000 (trained on WikiText-103 train only)
         -> CorpusArrays with the kind_map, stop and sym_freq of the WikiText-103 TRAIN split
            (copied, exactly as the corpus store copies them into every split) and targets = every
            position.

In causal mode a clause at position p reads only the tokens before p in the same document (the target
reads SYM_MASK, outside positions SYM_EDGE; tm_core.h), so position p's fired set describes the left
context t[0..p-1]. The appended END token makes position L (the END position) see the whole text;
position 0 sees nothing (its fired set is the same for every document). The clause representation of
a document is the sparse vector over clause ids

    x_j = #{p in the document : clause j fires at p}        (count variant)
    x_j = [x_j > 0]                                        (binary variant)

and the DF variant keeps only the K clauses with the highest document frequency over the unlabelled
training pool of the task (ties to the lower id).

Baselines built from the same token streams (text tokens only, the END token excluded):
bag of BPE unigrams, unigrams plus adjacent bigrams, the mean in-house CBOW input vector, and the bag
of exchange-cluster symbols of kind_map levels 1 to 3 (16, 243 and 2497 clusters at V 8000).

Everything here is CPU numpy/scipy; the GPU aggregation lives in the experiment script and is checked
against `aggregate_doc_counts` (the reference) and the engine's public fired_clauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

END_TEXT = " \n"


# ------------------------------------------------------------------ text
def wikitext_style(s: str) -> str:
    """Same rule as tmlm.cvtm.crtm.wikitext_style (tests check equality): a leading space, clitics
    n't, 's, 're, 've, 'll, 'd, 'm and a bare trailing ' split off, . , ? ! ; : split off, single spaces."""
    s = s.strip()
    s = re.sub(r"(\w)n't\b", r"\1 n't", s)
    s = re.sub(r"(\w)'(s|re|ve|ll|d|m)\b", r"\1 '\2", s)
    s = re.sub(r"(\w)'(\s|$)", r"\1 '\2", s)
    s = re.sub(r"([.,?!;:])", r" \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " " + s


def doc_text(s: str) -> str:
    """One downstream example as a document: WikiText surface form plus the END token text."""
    return wikitext_style(s) + END_TEXT


@dataclass(frozen=True)
class TokenDocs:
    """Token streams of a list of documents. tok u16 [N]; doc_start u32 [n_doc + 1]; the LAST token of
    every document is the END token (text tokens are doc_start[d] .. doc_start[d + 1] - 2)."""

    tok: np.ndarray
    doc_start: np.ndarray
    end_id: int

    @property
    def n_doc(self) -> int:
        return int(self.doc_start.shape[0] - 1)

    @property
    def n_tok(self) -> int:
        return int(self.tok.shape[0])

    def doc_of_pos(self) -> np.ndarray:
        ds = self.doc_start.astype(np.int64)
        return np.repeat(np.arange(self.n_doc, dtype=np.int64), np.diff(ds))

    def text_mask(self) -> np.ndarray:
        """bool [N]: True for text tokens, False for the END token of each document."""
        m = np.ones(self.n_tok, dtype=bool)
        m[self.doc_start[1:].astype(np.int64) - 1] = False
        return m


def encode(tokenizer, texts: Sequence[str]) -> TokenDocs:
    """BPE-encode the documents doc_text(t) for t in texts; asserts that every document ends with the
    END token (the tokenizer encodes " \\n" as one token after any text that ends in a non-space)."""
    from ..data.bpe import encode_docs

    end = tokenizer.encode(END_TEXT).ids
    if len(end) != 1:
        raise ValueError(f"END text must be one token, got {end}")
    tok, ds = encode_docs(tokenizer, [doc_text(t) for t in texts])
    last = tok[ds[1:].astype(np.int64) - 1]
    if not np.all(last == end[0]) or np.any(np.diff(ds.astype(np.int64)) < 2):
        raise ValueError("a document does not end with the END token or has no text token")
    return TokenDocs(tok=tok, doc_start=ds, end_id=int(end[0]))


def concat(parts: Sequence[TokenDocs]) -> tuple[TokenDocs, np.ndarray]:
    """Concatenate token docs; returns (all, doc_offset [len(parts) + 1])."""
    toks, starts, offs = [], [np.zeros(1, np.int64)], [0]
    base = 0
    for p in parts:
        if p.end_id != parts[0].end_id:
            raise ValueError("different END ids")
        toks.append(p.tok)
        starts.append(p.doc_start[1:].astype(np.int64) + base)
        base += p.n_tok
        offs.append(offs[-1] + p.n_doc)
    ds = np.concatenate(starts)
    if ds[-1] > 0xFFFFFFFF:
        raise OverflowError("N_tok does not fit u32")
    return TokenDocs(tok=np.concatenate(toks).astype(np.uint16), doc_start=ds.astype(np.uint32),
                     end_id=parts[0].end_id), np.asarray(offs, np.int64)


def corpus_arrays(docs: TokenDocs, ref):
    """CorpusArrays of the documents with the TRAIN-split statistics (kind_map, stop, sym_freq) of the
    reference split `ref` (a CorpusArrays of the WikiText-103 store) and targets = every position."""
    from ..engine.api import CorpusArrays

    return CorpusArrays(tok=np.ascontiguousarray(docs.tok, dtype=np.uint16),
                        doc_start=np.ascontiguousarray(docs.doc_start, dtype=np.uint32),
                        kind_map=np.array(ref.kind_map, dtype=np.uint16),
                        stop=np.array(ref.stop, dtype=np.uint8),
                        sym_freq=np.array(ref.sym_freq, dtype=np.uint32),
                        targets=np.arange(docs.n_tok, dtype=np.uint32))


# ------------------------------------------------------------------ sparse per-document vectors
@dataclass
class SparseDocs:
    """CSR over documents: row d holds ids[indptr[d]:indptr[d + 1]] (ascending) with counts cnt."""

    indptr: np.ndarray   # int64 [n_doc + 1]
    ids: np.ndarray      # uint32 [nnz]
    cnt: np.ndarray      # uint32 [nnz]
    n_cols: int

    @property
    def n_doc(self) -> int:
        return int(self.indptr.shape[0] - 1)

    def rows(self, idx: np.ndarray) -> "SparseDocs":
        idx = np.asarray(idx, np.int64)
        lo, hi = self.indptr[idx], self.indptr[idx + 1]
        lens = hi - lo
        ip = np.zeros(idx.shape[0] + 1, np.int64)
        ip[1:] = np.cumsum(lens)
        take = np.repeat(lo - ip[:-1], lens) + np.arange(ip[-1], dtype=np.int64)
        return SparseDocs(ip, self.ids[take], self.cnt[take], self.n_cols)

    def to_csr(self, value: str = "log1p", cols: np.ndarray | None = None, l2: bool = True):
        """scipy CSR (float64). value: 'count', 'log1p' (log(1 + count)) or 'binary'. cols: keep only
        these column ids, renumbered 0..len(cols)-1 in the given order. l2: unit rows (empty rows stay 0)."""
        import scipy.sparse as sp

        ids = self.ids.astype(np.int64)
        cnt = self.cnt.astype(np.float64)
        ip = self.indptr
        n_cols = self.n_cols
        if cols is not None:
            remap = np.full(self.n_cols, -1, np.int64)
            remap[np.asarray(cols, np.int64)] = np.arange(len(cols), dtype=np.int64)
            new = remap[ids]
            keep = new >= 0
            row = np.repeat(np.arange(self.n_doc, dtype=np.int64), np.diff(ip))[keep]
            ids, cnt = new[keep], cnt[keep]
            ip = np.zeros(self.n_doc + 1, np.int64)
            ip[1:] = np.cumsum(np.bincount(row, minlength=self.n_doc))
            n_cols = len(cols)
        if value == "log1p":
            data = np.log1p(cnt)
        elif value == "binary":
            data = np.ones_like(cnt)
        elif value == "count":
            data = cnt
        else:
            raise ValueError(value)
        X = sp.csr_matrix((data, ids, ip), shape=(self.n_doc, n_cols))
        X.sort_indices()
        if l2:
            nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            nrm[nrm == 0] = 1.0
            X = sp.diags(1.0 / nrm) @ X
            X = X.tocsr()
        return X

    def doc_freq_range(self, a: int, b: int) -> np.ndarray:
        """int64 [n_cols]: document frequency over the CONTIGUOUS rows a..b-1 (no copy of the rows)."""
        seg = self.ids[int(self.indptr[a]):int(self.indptr[b])]
        return np.bincount(seg.astype(np.int64), minlength=self.n_cols).astype(np.int64)

    def doc_freq(self, rows: np.ndarray | None = None) -> np.ndarray:
        """int64 [n_cols]: number of documents (of `rows`, default all) where the column is non-zero."""
        sub = self if rows is None else self.rows(rows)
        return np.bincount(sub.ids.astype(np.int64), minlength=self.n_cols).astype(np.int64)


def top_df_columns(df: np.ndarray, k: int) -> np.ndarray:
    """The k columns of highest document frequency (ties to the lower id), ascending ids; columns with
    df 0 are never selected."""
    df = np.asarray(df, np.int64)
    order = np.lexsort((np.arange(df.shape[0]), -df))
    top = order[: int(k)]
    top = top[df[top] > 0]
    return np.sort(top)


def aggregate_doc_counts(doc_of_event: np.ndarray, clause_of_event: np.ndarray, n_doc: int, n_cols: int) -> SparseDocs:
    """Reference aggregation: every (document, clause) event counted once per firing position."""
    d = np.asarray(doc_of_event, np.int64)
    j = np.asarray(clause_of_event, np.int64)
    if d.size and (d.min() < 0 or d.max() >= n_doc or j.min() < 0 or j.max() >= n_cols):
        raise ValueError("event outside the document or column range")
    key = d * n_cols + j
    u, c = np.unique(key, return_counts=True)
    dd = u // n_cols
    ip = np.zeros(n_doc + 1, np.int64)
    ip[1:] = np.cumsum(np.bincount(dd, minlength=n_doc))
    return SparseDocs(ip, (u % n_cols).astype(np.uint32), c.astype(np.uint32), int(n_cols))


def merge_rows(parts: Sequence[SparseDocs]) -> SparseDocs:
    """Stack SparseDocs blocks of consecutive documents (ids u32; counts keep the widest part dtype)."""
    ips = [np.zeros(1, np.int64)]
    base = 0
    for p in parts:
        ips.append(p.indptr[1:] + base)
        base += int(p.indptr[-1])
    cdt = np.result_type(*[p.cnt.dtype for p in parts])
    ids = np.empty(base, np.uint32)
    cnt = np.empty(base, cdt)
    o = 0
    for p in parts:
        n = int(p.indptr[-1])
        ids[o:o + n] = p.ids[:n]
        cnt[o:o + n] = p.cnt[:n]
        o += n
    return SparseDocs(np.concatenate(ips), ids, cnt, parts[0].n_cols)


# ------------------------------------------------------------------ baseline representations
def bag_of_tokens(docs: TokenDocs, vocab: int, bigrams: bool = False) -> SparseDocs:
    """Text tokens only (END excluded). Unigram column = token id; bigram (a, b) of ADJACENT text tokens
    column = vocab + a * vocab + b (sparse, n_cols = vocab + vocab^2)."""
    tm = docs.text_mask()
    dop = docs.doc_of_pos()
    tok = docs.tok.astype(np.int64)
    d_ev = [dop[tm]]
    c_ev = [tok[tm]]
    n_cols = vocab
    if bigrams:
        a = np.arange(docs.n_tok - 1)
        ok = tm[a] & tm[a + 1] & (dop[a] == dop[a + 1])
        d_ev.append(dop[a[ok]])
        c_ev.append(vocab + tok[a[ok]] * vocab + tok[a[ok] + 1])
        n_cols = vocab + vocab * vocab
    return _aggregate_wide(np.concatenate(d_ev), np.concatenate(c_ev), docs.n_doc, n_cols)


def _aggregate_wide(d: np.ndarray, c: np.ndarray, n_doc: int, n_cols: int) -> SparseDocs:
    """aggregate_doc_counts for column spaces too wide for d * n_cols to fit int64 comfortably."""
    order = np.lexsort((c, d))
    d, c = d[order], c[order]
    brk = np.flatnonzero((d[1:] != d[:-1]) | (c[1:] != c[:-1])) + 1
    starts = np.concatenate([[0], brk]) if d.size else np.zeros(0, np.int64)
    cnt = np.diff(np.concatenate([starts, [d.size]])) if d.size else np.zeros(0, np.int64)
    ud, uc = d[starts], c[starts]
    ip = np.zeros(n_doc + 1, np.int64)
    ip[1:] = np.cumsum(np.bincount(ud, minlength=n_doc))
    if n_cols > 0xFFFFFFFF:
        raise OverflowError("column ids must fit u32")
    return SparseDocs(ip, uc.astype(np.uint32), cnt.astype(np.uint32), int(n_cols))


def compact_columns(sd: SparseDocs) -> tuple[SparseDocs, np.ndarray]:
    """Renumber the used columns 0..U-1 (ascending original ids); returns (compact, original ids)."""
    used, inv = np.unique(np.asarray(sd.ids, np.int64), return_inverse=True)
    return SparseDocs(sd.indptr.copy(), inv.reshape(-1).astype(np.uint32), sd.cnt.copy(), int(used.shape[0])), used


def cluster_bags(docs: TokenDocs, kind_map: np.ndarray, levels: Sequence[int] = (1, 2, 3)) -> tuple[SparseDocs, np.ndarray]:
    """Bag of exchange-cluster symbols of the text tokens: one column per (level, cluster symbol).
    Returns (bags over compact columns, the symbol id of every column)."""
    km = np.asarray(kind_map, np.int64)
    tm = docs.text_mask()
    dop = docs.doc_of_pos()[tm]
    tok = docs.tok.astype(np.int64)[tm]
    d = np.concatenate([dop] * len(levels))
    c = np.concatenate([km[k][tok] for k in levels])
    sd = _aggregate_wide(d, c, docs.n_doc, int(km.max()) + 1)
    return compact_columns(sd)


def mean_embedding(docs: TokenDocs, W: np.ndarray) -> np.ndarray:
    """float64 [n_doc, dim]: mean of W[t] over the text tokens of every document."""
    tm = docs.text_mask()
    dop = docs.doc_of_pos()[tm]
    tok = docs.tok.astype(np.int64)[tm]
    W = np.asarray(W, np.float64)
    out = np.zeros((docs.n_doc, W.shape[1]))
    np.add.at(out, dop, W[tok])
    n = np.bincount(dop, minlength=docs.n_doc).astype(np.float64)
    return out / np.maximum(n, 1.0)[:, None]


def thermometer(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Binarise dense real features: bit (i, b) = X[:, i] >= edges[i, b]. edges [dim, n_bits] (quantiles
    of the training rows). Returns uint32 [n, dim * n_bits]."""
    X = np.asarray(X, np.float64)
    out = (X[:, :, None] >= edges[None, :, :]).reshape(X.shape[0], -1)
    return out.astype(np.uint32)


def quantile_edges(X: np.ndarray, n_bits: int) -> np.ndarray:
    qs = (np.arange(1, n_bits + 1) / (n_bits + 1.0))
    return np.quantile(np.asarray(X, np.float64), qs, axis=0).T.copy()
