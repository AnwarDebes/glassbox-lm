"""Images as token documents for the docs/11 engine: Fashion-MNIST patch tokens (docs/13 A7, phase R4, H17.10).

Data: the Fashion-MNIST TRAIN split only (60,000 images, Hugging Face `zalando-datasets/fashion_mnist` at the
pinned revision below, MIT licence); the test split is never downloaded or read. A hash-carved dev set of exactly
DEV_SIZE = 10,000 images (the image row indices i with the 10,000 smallest mix64(i XOR DEV_SALT), docs/11 section
3 mix64) is held out of every unsupervised statistic (codebook, clusters, stop list, symbol frequencies) and of
self-supervised training; the other 50,000 images are the POOL.

Tokenization (one image = one document):
    binarise        pixel > 75 -> 1 (THRESH)
    patches         7 x 7 non-overlapping 4 x 4 patches; the 16-bit pattern of a patch has bit 4*dy + dx set
                    when pixel (4r + dy, 4c + dx) is on (dy, dx in 0..3)
    codebook        the N_CODES = 1,999 most frequent patterns over the POOL patches get token ids 0..1998 in
                    order of (count descending, pattern value ascending); a pattern outside the codebook maps to
                    the codebook entry of smallest Hamming distance, ties to the LOWER token id (the more
                    frequent pattern); a 65,536-entry lookup table holds the map
    row end         token ROW_END = 1,999 after every row of 7 patches, so V = 2,000 and the offset -8 / +8
                    literals of the engine (docs/11 section 1.4) read the patch directly above / below
    document        7 rows x (7 patches + ROW_END) = 56 tokens, in row-major order
    stop list       the N_STOP = 64 most frequent POOL tokens (tmlm.data.synthetic.stop_of, as for text; ROW_END
                    is one of them whenever it ranks in the top 64)
    kind_map        the repository's 3-level 16-way exchange clustering on the POOL bigram counts
                    (tmlm.data.clusters.bigram_counts, hierarchical_exchange, kind_map_from_levels), run as the
                    text store runs it (experiments/m2_build_data.py step_train_stats: cluster seed 1, 20 iterations)
    sym_freq        tmlm.data.synthetic.sym_freq_of over the POOL tokens
    targets         every PATCH position (never a ROW_END position) of every POOL image

Store: the engine's corpus format (tmlm.data.corpus_store): data/<corpus>/bpe<V>/train/{tok, doc_start, kind_map,
stop, sym_freq, targets}.npy plus labels.npy (u8 [60000]), dev.npy (u8 [60000], 1 = dev), codebook.npy
(u16 [1999]), lut.npy (u16 [65536]) and meta.json with the sha256 of every array. `bpe<V>` is the store's fixed
directory name for a V-token store (here patch codes, not BPE). Files are only created, never overwritten
(protocol rule 9): an existing store is re-read and must hash equal to a fresh build.

Also here: the F1 frequency-trap lever of A7 (target subsampling by mix64 of the position), the quadrant of a
patch, the bag of visual words per quadrant, a random kind_map (the "concepts removed" control), the frozen
hidden bitmap of the masked-prediction diagnostic, and stratified label draws for the few-label probes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

PATCH = 4
GRID = 7
ROW_LEN = GRID + 1
TOKENS_PER_IMAGE = GRID * ROW_LEN  # 56
N_CODES = 1999
VOCAB = N_CODES + 1
THRESH = 75
N_STOP = 64
N_IMAGES = 60000
DEV_SIZE = 10000
DEV_SALT = 0xFA5410A7D3
HIDE_SALT = 0x51DE0A7F
HIDE_RATE = 0.15
F1_T = 1e-4
CORPUS = "fashion_mnist_p4"
SPLIT = "train"
HF = {"repo": "zalando-datasets/fashion_mnist", "revision": "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2",
      "file": "fashion_mnist/train-00000-of-00001.parquet",
      "lfs_sha256": "a5769a70af055b15ada8bd3dad9057557554fdd30e17cba45e7ef0f8bf79c5bb", "bytes": 30931277,
      "license": "MIT (dataset card: license: mit)"}
EXTRA_ARRAYS = ("labels", "dev", "codebook", "lut")
CORE_ARRAYS = ("tok", "doc_start", "kind_map", "stop", "sym_freq", "targets")
SEED_TAG = (20260924, 1710)


# ------------------------------------------------------------------ hashing (docs/11 section 3)
def mix64_np(z) -> np.ndarray:
    """Vectorised tmlm.engine.rng.mix64 over uint64 (wraps mod 2^64)."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def q32_np(p: np.ndarray) -> np.ndarray:
    """Q32 thresholds min(2^32 - 1, floor(p * 2^32)) as uint64 (p clipped to [0, 1])."""
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    return np.minimum(np.floor(p * 4294967296.0), 4294967295.0).astype(np.uint64)


def dev_mask(n_total: int = N_IMAGES, dev_size: int = DEV_SIZE, salt: int = DEV_SALT) -> np.ndarray:
    """bool [n_total]: the dev_size row indices with the smallest mix64(i XOR salt) (mix64 is a bijection, no ties)."""
    h = mix64_np(np.arange(n_total, dtype=np.uint64) ^ np.uint64(salt))
    dev = np.zeros(n_total, dtype=bool)
    dev[np.argsort(h, kind="stable")[:dev_size]] = True
    return dev


# ------------------------------------------------------------------ images -> patterns -> tokens
def read_fashion_parquet(path: str | Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(images u8 [n, 28, 28], labels u8 [n]) of the train parquet file in row order (first `limit` rows)."""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(str(path), columns=["image", "label"])
    n = table.num_rows if limit is None else min(int(limit), table.num_rows)
    table = table.slice(0, n)
    imgs = table.column("image").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    out = np.empty((n, 28, 28), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        a = np.asarray(Image.open(io.BytesIO(rec["bytes"])))
        if a.shape != (28, 28) or a.dtype != np.uint8:
            raise ValueError(f"row {i}: unexpected image {a.shape} {a.dtype}")
        out[i] = a
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("labels outside 0..9")
    return out, labels.astype(np.uint8)


def binarize(images: np.ndarray, thresh: int = THRESH) -> np.ndarray:
    return (np.asarray(images) > thresh).astype(np.uint8)


def patch_patterns(bits: np.ndarray) -> np.ndarray:
    """u16 [n, 7, 7]: pattern of patch (r, c) = sum over dy, dx of bits[4r+dy, 4c+dx] << (4*dy + dx)."""
    b = np.asarray(bits, dtype=np.uint16).reshape(-1, GRID, PATCH, GRID, PATCH)  # n, r, dy, c, dx
    w = (np.uint16(1) << np.arange(PATCH * PATCH, dtype=np.uint16)).reshape(PATCH, PATCH)  # [dy, dx]
    return (b * w[None, None, :, None, :]).sum(axis=(2, 4), dtype=np.uint32).astype(np.uint16)


def pattern_counts(patterns: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(patterns, np.int64).reshape(-1), minlength=1 << 16).astype(np.int64)


def build_codebook(counts: np.ndarray, n_codes: int = N_CODES) -> np.ndarray:
    """u16 [n_codes]: patterns ranked by (count descending, pattern ascending); token id = rank."""
    counts = np.asarray(counts, np.int64)
    order = np.lexsort((np.arange(counts.shape[0]), -counts))
    return order[:n_codes].astype(np.uint16)


_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def nearest_code_lut(codebook: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """u16 [65536]: token id of every pattern; exact for codebook patterns, else the smallest Hamming distance
    with ties to the lower token id (argmin returns the first minimum)."""
    cb = np.asarray(codebook, np.uint32)
    lut = np.empty(1 << 16, dtype=np.uint16)
    pats = np.arange(1 << 16, dtype=np.uint32)
    for lo in range(0, pats.shape[0], chunk):
        d = _POP16[pats[lo:lo + chunk, None] ^ cb[None, :]]
        lut[lo:lo + chunk] = np.argmin(d, axis=1).astype(np.uint16)
    return lut


def image_tokens(patterns: np.ndarray, lut: np.ndarray, row_end: int) -> np.ndarray:
    """u16 [n * 56]: per image, per row, the 7 patch tokens then row_end."""
    t = np.asarray(lut)[np.asarray(patterns, np.int64)]  # n, 7, 7
    out = np.full((t.shape[0], GRID, ROW_LEN), row_end, dtype=np.uint16)
    out[:, :, :GRID] = t
    return out.reshape(-1)


def doc_starts(n_images: int) -> np.ndarray:
    return (np.arange(n_images + 1, dtype=np.uint64) * TOKENS_PER_IMAGE).astype(np.uint32)


# ------------------------------------------------------------------ position geometry
def position_rc(n_tok: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image, row, col) of every token position (col 7 = ROW_END)."""
    p = np.arange(n_tok, dtype=np.int64)
    o = p % TOKENS_PER_IMAGE
    return p // TOKENS_PER_IMAGE, o // ROW_LEN, o % ROW_LEN


def quadrant(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right; the middle patch row and column (index 3, pixels
    12..15) belong to the top and left halves (fixed rule). Sizes 16, 12, 12, 9."""
    return (2 * (np.asarray(row) >= 4) + (np.asarray(col) >= 4)).astype(np.int64)


def patch_positions(image_ids: np.ndarray) -> np.ndarray:
    """u32 positions of all 49 patches of the given images, image by image, row-major."""
    img = np.asarray(image_ids, np.int64)
    off = np.array([r * ROW_LEN + c for r in range(GRID) for c in range(GRID)], np.int64)
    return (img[:, None] * TOKENS_PER_IMAGE + off[None, :]).reshape(-1).astype(np.uint32)


def frozen_hidden(n_tok: int, rate: float = HIDE_RATE, salt: int = HIDE_SALT) -> np.ndarray:
    """u8 [n_tok] hidden bitmap of the masked-prediction diagnostic: (mix64(p XOR salt) >> 32) < q32(rate)."""
    u = mix64_np(np.arange(n_tok, dtype=np.uint64) ^ np.uint64(salt)) >> np.uint64(32)
    return (u < q32_np(rate)).astype(np.uint8)


# ------------------------------------------------------------------ the store
def build_store_arrays(images: np.ndarray, labels: np.ndarray, dev: np.ndarray, n_codes: int = N_CODES,
                       cluster_seed: int = 1, cluster_max_iter: int = 20, n_stop: int = N_STOP) -> tuple[dict, dict]:
    """All store arrays and a meta dict (cluster logs, counts). `dev` bool [n] marks the held-out images."""
    from ..data import clusters
    from ..data.synthetic import stop_of, sym_freq_of

    images = np.asarray(images)
    dev = np.asarray(dev, dtype=bool)
    n = images.shape[0]
    if dev.shape != (n,) or np.asarray(labels).shape != (n,):
        raise ValueError("labels and dev must have one entry per image")
    V = n_codes + 1
    pats = patch_patterns(binarize(images))
    pool = np.flatnonzero(~dev)
    counts = pattern_counts(pats[pool])
    codebook = build_codebook(counts, n_codes)
    lut = nearest_code_lut(codebook)
    tok = image_tokens(pats, lut, row_end=n_codes)
    ds = doc_starts(n)
    pool_tok = tok.reshape(n, TOKENS_PER_IMAGE)[pool].reshape(-1)
    pool_ds = doc_starts(pool.shape[0])
    B = clusters.bigram_counts(pool_tok, pool_ds, V)
    levels, logs = clusters.hierarchical_exchange(B, seed=cluster_seed, max_iter=cluster_max_iter)
    if not clusters.check_refinement(levels):
        raise RuntimeError("cluster levels do not refine")
    km = clusters.kind_map_from_levels(V, levels)
    stop = stop_of(pool_tok, V, n_stop)
    sym_freq = sym_freq_of(pool_tok, km, V)
    targets = patch_positions(pool)
    arrays = {"tok": tok, "doc_start": ds, "kind_map": km, "stop": stop, "sym_freq": sym_freq, "targets": targets,
              "labels": np.asarray(labels, np.uint8), "dev": dev.astype(np.uint8), "codebook": codebook, "lut": lut}
    n_distinct = int((counts > 0).sum())
    covered = int(counts[codebook.astype(np.int64)].sum())
    meta = {"n_images": int(n), "n_pool": int(pool.shape[0]), "n_dev": int(dev.sum()), "vocab": int(V),
            "n_codes": int(n_codes), "row_end": int(n_codes), "distinct_patterns_pool": n_distinct,
            "pool_patches": int(pats[pool].size), "pool_patches_exact_in_codebook": covered,
            "pool_patch_coverage": covered / max(1, int(pats[pool].size)),
            "row_end_is_stop": bool(stop[n_codes]), "n_targets": int(targets.shape[0]),
            "bigram_types": int(B.nnz),
            "clusters": {"seed": int(cluster_seed), "max_iter": int(cluster_max_iter),
                         "method": "3-level 16-way exchange, POOL bigrams (tmlm.data.clusters)",
                         **{f"level{lg.level}": {**lg.as_dict(), **clusters.cluster_stats(levels[lg.level - 1], lg.n_clusters)} for lg in logs}}}
    return arrays, meta


def array_hashes(arrays: dict) -> dict:
    from ..common import hashing

    return {f"{k}.npy": hashing.sha256_array(np.ascontiguousarray(v)) for k, v in sorted(arrays.items())}


def write_store(arrays: dict, meta: dict, corpus: str = CORPUS, root: Path | None = None) -> dict:
    """Create the store (never overwrite). If it exists, every recorded hash must equal the fresh build's."""
    from ..data import corpus_store

    V = int(arrays["kind_map"].shape[1])
    d = corpus_store.split_dir(corpus, V, SPLIT, root)
    hashes = array_hashes(arrays)
    if (d / "meta.json").exists():
        old = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if old.get("sha256") != hashes:
            bad = sorted(k for k in hashes if old.get("sha256", {}).get(k) != hashes[k])
            raise RuntimeError(f"{d}: existing store differs from a fresh build in {bad}; not overwritten")
        return old
    corpus_store.write_split_arrays(corpus, V, SPLIT, {k: arrays[k] for k in CORE_ARRAYS}, root)
    for k in EXTRA_ARRAYS:
        corpus_store.write_array(d / f"{k}.npy", arrays[k])
    full = {"corpus": corpus, "vocab": V, "split": SPLIT, "n_tok": int(arrays["tok"].shape[0]),
            "n_doc": int(arrays["doc_start"].shape[0] - 1), "n_stop": int(arrays["stop"].sum()),
            "tokens_per_image": TOKENS_PER_IMAGE, "sha256": hashes, **meta}
    corpus_store.write_split_meta(corpus, V, SPLIT, full, root)
    return full


def load_store(corpus: str = CORPUS, vocab: int = VOCAB, root: Path | None = None, check: bool = True):
    """(CorpusArrays, extras {labels, dev, codebook, lut}, meta). check=True re-hashes every array."""
    from ..common import hashing
    from ..data import corpus_store

    ca = corpus_store.load_split(corpus, vocab, SPLIT, root, mmap=False, check=check)
    d = corpus_store.split_dir(corpus, vocab, SPLIT, root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    extras = {k: np.load(d / f"{k}.npy") for k in EXTRA_ARRAYS}
    if check:
        for k, v in extras.items():
            if hashing.sha256_array(v) != meta["sha256"][f"{k}.npy"]:
                raise ValueError(f"{d}/{k}.npy: sha256 mismatch with meta.json")
    return ca, extras, meta


# ------------------------------------------------------------------ levers and controls
def f1_keep(positions: np.ndarray, classes: np.ndarray, t: float = F1_T) -> np.ndarray:
    """A7 lever F1: keep target p (class y) w.p. min(1, sqrt(t / f(y)) + t / f(y)), f(y) = relative frequency of y
    among the given targets; the draw is (mix64(p) >> 32) < q32(prob) (no seed: the same subsample for every
    run). Returns bool [n]."""
    pos = np.asarray(positions, np.uint64)
    cls = np.asarray(classes, np.int64)
    cnt = np.bincount(cls, minlength=int(cls.max()) + 1 if cls.size else 1).astype(np.float64)
    f = cnt[cls] / max(1, cls.shape[0])
    prob = np.minimum(1.0, np.sqrt(t / f) + t / f)
    u = mix64_np(pos) >> np.uint64(32)
    return (prob >= 1.0) | (u < q32_np(prob))


def random_kind_map(kind_map: np.ndarray, seed: int) -> np.ndarray:
    """Control "concepts removed": token t takes the cluster symbols of token perm[t] (perm uniform from
    SeedSequence([20260924, 1710, 7, seed])); row 0 stays the identity, so cluster sizes and the 3-level
    refinement are kept and only the membership is random."""
    km = np.asarray(kind_map)
    perm = np.random.default_rng(np.random.SeedSequence([*SEED_TAG, 7, int(seed)])).permutation(km.shape[1])
    out = km.copy()
    out[1:] = km[1:, perm]
    return out


def stratified_sample(labels: np.ndarray, pool_idx: np.ndarray, per_class: int, key: Sequence[int]) -> np.ndarray:
    """per_class indices of every class (ascending class order) drawn uniformly without replacement from pool_idx
    by default_rng(SeedSequence([20260924, 1710, *key])); returned sorted."""
    labels = np.asarray(labels)
    pool_idx = np.sort(np.asarray(pool_idx, np.int64))
    rng = np.random.default_rng(np.random.SeedSequence([*SEED_TAG, *[int(k) for k in key]]))
    out = []
    for c in np.unique(labels[pool_idx]):
        members = pool_idx[labels[pool_idx] == c]
        if members.shape[0] < per_class:
            raise ValueError(f"class {c}: {members.shape[0]} pool members < {per_class}")
        out.append(rng.choice(members, size=per_class, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


def bovw_quadrant(tok: np.ndarray, image_ids: np.ndarray, vocab: int = VOCAB):
    """Bag of visual words per quadrant of the given images: tmlm.ssl.features.SparseDocs with one row per image
    (in the given order) and column code * 4 + quadrant, counts over the 49 patch positions."""
    from .features import aggregate_doc_counts

    img = np.asarray(image_ids, np.int64)
    pos = patch_positions(img).astype(np.int64)
    o = pos % TOKENS_PER_IMAGE
    q = quadrant(o // ROW_LEN, o % ROW_LEN)
    col = np.asarray(tok, np.int64)[pos] * 4 + q
    row = np.repeat(np.arange(img.shape[0], dtype=np.int64), GRID * GRID)
    return aggregate_doc_counts(row, col, img.shape[0], vocab * 4)


def image_code_from_events(pos_of_event: np.ndarray, clause_of_event: np.ndarray, image_row: dict | np.ndarray,
                           n_rows: int, n_clauses: int):
    """Reference aggregation of fired (position, clause) events into image codes: SparseDocs with row
    image_row[image of p] and column clause * 4 + quadrant(p)."""
    from .features import aggregate_doc_counts

    p = np.asarray(pos_of_event, np.int64)
    img = p // TOKENS_PER_IMAGE
    o = p % TOKENS_PER_IMAGE
    q = quadrant(o // ROW_LEN, o % ROW_LEN)
    rows = np.asarray(image_row)[img]
    return aggregate_doc_counts(rows, np.asarray(clause_of_event, np.int64) * 4 + q, n_rows, n_clauses * 4)
