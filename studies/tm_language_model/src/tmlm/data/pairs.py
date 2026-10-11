"""The proposal's paragraph-pair task (docs/05 E6 tier 1, H1, H2; docs/06; docs/13 section 4).

Paragraphs: inside one article, a block is a maximal run of non-blank lines of the raw text
that are not heading lines (a heading line starts and ends with " = " after the newline is
removed, at any level). Blocks are tokenized with the BPE-8000 tokenizer; blocks with fewer
than MIN_TOKENS = 20 tokens are dropped.

Pairs: a positive pair is two DISTINCT paragraphs of the same article (label 1); a negative
pair is one paragraph from each of two DIFFERENT articles of the same split (label 0). Pairs
are built INSIDE one split only, balanced 50/50, without duplicates (a pair is a set of two
(article, paragraph) ids). Sampling is uniform over eligible articles (positives: articles with
at least two paragraphs; negatives: two distinct articles with at least one) and then uniform
over their paragraphs, from a numpy Generator seeded with SeedSequence([BASE_SEED, split code]).
When fewer distinct positive pairs exist than requested, every positive pair is enumerated and
the negatives are cut to the same number (balance is kept and meta.json states it).

Storage: <corpus>/pairs_bpe<V>/<split>.npz with a_ptr/a_tok and b_ptr/b_tok (CSR-style
ragged token arrays, u32 pointers, u16 ids), a_doc, b_doc (article ids in the split), a_para,
b_para (block index inside the article) and label (u8), plus <split>.meta.json with hashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import hashing
from .corpus_store import data_root, write_new_json
from .evalsets import SPLIT_CODE, combined_hash, record_test_access

MIN_TOKENS = 20
PAIR_BASE_SEED = 20260923


def is_heading_line(line: str) -> bool:
    s = line.rstrip("\n")
    return s.startswith(" = ") and s.endswith(" = ")


def blocks_of_article(text: str) -> list[str]:
    """Maximal runs of non-blank, non-heading lines; each block keeps its line breaks."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.strip() or is_heading_line(line):
            if current:
                blocks.append("".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks


def paragraphs_of_split(texts: list[str], tok, min_tokens: int = MIN_TOKENS) -> list[list[np.ndarray]]:
    """Per article: the token arrays (u16) of its kept blocks, in order."""
    from .bpe import _parallel_tokenizers

    per_article: list[list[str]] = [blocks_of_article(t) for t in texts]
    flat = [b for blocks in per_article for b in blocks]
    with _parallel_tokenizers():
        encs = tok.encode_batch(flat) if flat else []
    out: list[list[np.ndarray]] = []
    k = 0
    for blocks in per_article:
        kept = []
        for _ in blocks:
            ids = np.asarray(encs[k].ids, dtype=np.uint16)
            k += 1
            if ids.shape[0] >= min_tokens:
                kept.append(ids)
        out.append(kept)
    return out


def build_pairs(paragraphs: list[list[np.ndarray]], n_pairs: int, split: str, seed_base: int = PAIR_BASE_SEED) -> dict:
    """Sample balanced pairs (see the module docstring). Returns arrays plus a `rule` string."""
    rng = np.random.default_rng(np.random.SeedSequence([seed_base, SPLIT_CODE[split]]))
    counts = np.array([len(p) for p in paragraphs], dtype=np.int64)
    pos_articles = np.flatnonzero(counts >= 2)
    any_articles = np.flatnonzero(counts >= 1)
    n_pos_possible = int((counts[pos_articles] * (counts[pos_articles] - 1) // 2).sum())
    n_pos_wanted = n_pairs // 2
    rule = "sampled"
    pos: list[tuple[int, int, int, int]] = []
    if n_pos_possible <= n_pos_wanted:
        rule = f"all {n_pos_possible} distinct positive pairs enumerated (fewer than requested), negatives cut to match"
        for d in pos_articles:
            for i in range(counts[d]):
                for j in range(i + 1, counts[d]):
                    pos.append((int(d), i, int(d), j))
    else:
        seen: set[tuple[int, int, int]] = set()
        tries = 0
        while len(pos) < n_pos_wanted:
            tries += 1
            if tries > 50 * n_pos_wanted + 1000:
                raise RuntimeError("positive pair sampling did not converge")
            d = int(pos_articles[rng.integers(0, pos_articles.shape[0])])
            i, j = sorted(rng.choice(counts[d], size=2, replace=False).tolist())
            key = (d, i, j)
            if key in seen:
                continue
            seen.add(key)
            pos.append((d, i, d, j))
    n_neg_wanted = len(pos) if n_pos_possible <= n_pos_wanted else n_pairs - n_pos_wanted
    if any_articles.shape[0] < 2:
        raise ValueError("negative pairs need at least two articles with a paragraph")
    neg: list[tuple[int, int, int, int]] = []
    seen_neg: set[tuple[int, int, int, int]] = set()
    tries = 0
    while len(neg) < n_neg_wanted:
        tries += 1
        if tries > 50 * n_neg_wanted + 1000:
            raise RuntimeError("negative pair sampling did not converge")
        d1, d2 = any_articles[rng.choice(any_articles.shape[0], size=2, replace=False)].tolist()
        i = int(rng.integers(0, counts[d1]))
        j = int(rng.integers(0, counts[d2]))
        key = (d1, i, d2, j) if (d1, i) < (d2, j) else (d2, j, d1, i)
        if key in seen_neg:
            continue
        seen_neg.add(key)
        neg.append((int(d1), i, int(d2), j))
    allp = [(p, 1) for p in pos] + [(n, 0) for n in neg]
    order = rng.permutation(len(allp))
    a_doc = np.array([allp[k][0][0] for k in order], dtype=np.uint32)
    a_para = np.array([allp[k][0][1] for k in order], dtype=np.uint32)
    b_doc = np.array([allp[k][0][2] for k in order], dtype=np.uint32)
    b_para = np.array([allp[k][0][3] for k in order], dtype=np.uint32)
    label = np.array([allp[k][1] for k in order], dtype=np.uint8)

    def ragged(docs, paras):
        toks = [paragraphs[int(d)][int(i)] for d, i in zip(docs, paras)]
        ptr = np.zeros(len(toks) + 1, dtype=np.uint32)
        if toks:
            ptr[1:] = np.cumsum([t.shape[0] for t in toks])
        flat = np.concatenate(toks).astype(np.uint16) if toks else np.zeros(0, dtype=np.uint16)
        return ptr, flat

    a_ptr, a_tok = ragged(a_doc, a_para)
    b_ptr, b_tok = ragged(b_doc, b_para)
    return {"a_ptr": a_ptr, "a_tok": a_tok, "b_ptr": b_ptr, "b_tok": b_tok, "a_doc": a_doc, "b_doc": b_doc,
            "a_para": a_para, "b_para": b_para, "label": label, "rule": rule,
            "n_pos": len(pos), "n_neg": len(neg), "n_pos_possible": n_pos_possible,
            "n_articles_with_paragraphs": int(any_articles.shape[0]),
            "n_paragraphs": int(counts.sum())}


def pairs_dir(corpus: str, vocab: int, root: Path | None = None) -> Path:
    return (root or data_root()) / corpus / f"pairs_bpe{vocab}"


def save_pairs(corpus: str, vocab: int, split: str, built: dict, root: Path | None = None) -> dict:
    path = pairs_dir(corpus, vocab, root) / f"{split}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in built.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in built.items() if not isinstance(v, np.ndarray)}
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    hashes = {k: hashing.sha256_array(v) for k, v in arrays.items()}
    meta = {"corpus": corpus, "vocab": vocab, "split": split, "min_tokens": MIN_TOKENS,
            "seed": {"base": PAIR_BASE_SEED, "split_code": SPLIT_CODE[split]},
            "n_pairs": int(arrays["label"].shape[0]), **scalars, "sha256": hashes,
            "sha256_combined": combined_hash(hashes), "sha256_file": hashing.sha256_file(path)}
    write_new_json(path.with_suffix(".meta.json"), meta)
    return meta


def load_pairs(corpus: str, vocab: int, split: str, root: Path | None = None,
               purpose: str | None = None, check: bool = True) -> dict:
    if split == "test":
        if not purpose:
            raise ValueError("loading test pairs needs a purpose (the access is counted)")
        record_test_access("pairs.load_pairs", purpose, root=root)
    path = pairs_dir(corpus, vocab, root) / f"{split}.npz"
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if check:
        for k, v in arrays.items():
            if hashing.sha256_array(v) != meta["sha256"][k]:
                raise ValueError(f"{path}:{k}: sha256 mismatch with meta")
    return {**arrays, "meta": meta}
