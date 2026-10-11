"""R1 pretext tournament of docs/13 amendment A7 (family H17): the arms A0 to A6.

An arm is a docs/11 engine configuration plus data-level levers. Every lever is implemented through the
engine's EXAMPLE SET (`CorpusArrays.targets`, docs/11 section 2: "the epoch permutation runs over indices
into it and p = targets[perm(e, idx)]") or through the static tables of `CorpusArrays` (`kind_map`,
`sym_freq`), never by changing engine code:

    A0  default masked                       mode masked, every docs/11 section 7 default
    A1  masked + F1                          targets = the F1 subsample (fixed for all epochs)
    A2  F1 + F3 + F4                         A1 with neg_budget 1 (F3) and cand_tries 4 (F4)
    A3  F2                                   targets = at most M = 2,000 positions per class, redrawn per epoch
    A4  causal                               mode causal, defaults
    A5  diffusion                            mode diffusion, defaults
    A6  A1 with a random kind_map            A1 with the cluster levels of kind_map permuted over tokens

Common to every arm (A7 R1): V 8000, C 65,536, the first 20M WikiText-103 training tokens cut at a document
boundary (`slice_prefix`, the rule of experiments/m3_tiny_lm.py slice_train), 4 epochs.

F1 target subsampling (A7): keep target position p with probability min(1, sqrt(t / f(y)) + t / f(y)),
t = 1e-4, y = tok[p], draw by mix64(p). Reading used here (the A7 text leaves these two points open, both
fixed before any R1 output):
  - f(y) is the relative frequency of y in the training prefix itself (count / prefix tokens): the data the
    run sees. The docs/11 `sym_freq` table (whole train split) is not used for it.
  - The draw follows the docs/11 section 3 Bernoulli convention: keep iff (mix64(p) >> 32) < q32(prob),
    q32(x) = min(2^32 - 1, floor(x * 2^32)); a probability of at least 1 keeps p unconditionally (min(1, .)
    means "always"). mix64(p) has no seed and no epoch, so the subsample is the same for every seed and epoch.
F2 class cap (A7): at most M = 2,000 targets per class per epoch. Reading: per epoch e, a class with at most
  M positions keeps all of them; a larger class keeps the M positions with the smallest
  rng64(0, STREAM_F2, e, p, 0) (docs/11 section 3 rng64 with a host-only stream id; ties toward the lower
  position). Redrawing per epoch uses more of the data over 4 epochs than a fixed cap would, at the same cap.
  Because `CorpusArrays` is fixed at engine construction, a per-epoch target set means one engine instance
  per epoch that loads the previous state (`load_state`); the state is complete (docs/11 section 2), so this
  equals one engine with per-epoch targets (tests/test_ssl_r1_pretext.py checks it bit for bit against
  `train_explicit` with host-computed permutations).
F3 `neg_budget` 1 and F4 `cand_tries` 4 are EngineConfig fields (docs/11 sections 8b, 8d).
Random kind_map (A6): one fixed permutation pi of the token ids (ascending rng64(0, STREAM_KM, t, 0, 0), ties
  toward the lower id); kind_map'[k][t] = kind_map[k][pi[t]] for k = 1..3, kind_map'[0] = identity. Cluster
  sizes, the prefix hierarchy of the three levels and the number of symbols are kept; which tokens share a
  cluster becomes random (the concepts are removed from the literals). `sym_freq` is recomputed for the new
  map from the same token counts (`random_kind_map`).

Everything here is host-side numpy; nothing depends on thread layout or batch composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..engine.api import CorpusArrays
from ..engine.config import EngineConfig, n_sym, sym_edge, sym_mask

VOCAB = 8000
N_CLAUSES = 65_536
N_TOKENS = 20_000_000
EPOCHS = 4
F1_T = 1e-4
F2_M = 2000
# Host-only stream ids for the data levers (the engine's streams are 1..16, docs/11 section 3).
STREAM_F2 = 0xA7F2
STREAM_KM = 0xA7C6

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_STREAM_MUL = 0xD6E8FEB86659FD93
Q32_ONE = (1 << 32) - 1


# ------------------------------------------------------------------ randomness (docs/11 section 3, vectorised)
def mix64_np(z) -> np.ndarray:
    """docs/11 mix64 over a uint64 array (arithmetic wraps mod 2^64; equals tmlm.engine.rng.mix64)."""
    z = np.asarray(z, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        z += _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
    return z ^ (z >> np.uint64(31))


def rng64_np(seed: int, stream: int, a, b, c) -> np.ndarray:
    """docs/11 rng64(seed, stream, a, b, c) with array arguments (broadcast); equals tmlm.engine.rng.rng64."""
    s0 = (int(seed) ^ ((int(stream) * _STREAM_MUL) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    z = mix64_np(np.uint64(s0))
    with np.errstate(over="ignore"):
        z = mix64_np(z + np.asarray(a, dtype=np.uint64))
        z = mix64_np(z + np.asarray(b, dtype=np.uint64))
        return mix64_np(z + np.asarray(c, dtype=np.uint64))


def q32(p: float) -> int:
    """docs/11 host conversion: min(2^32 - 1, floor(p * 2^32)) for p in [0, 1]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(p)
    return min(Q32_ONE, int(p * 4294967296.0))


# ------------------------------------------------------------------ arms
@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    f1: bool = False
    f2: bool = False
    random_kind_map: bool = False
    engine: tuple[tuple[str, Any], ...] = field(default_factory=tuple)  # EngineConfig overrides (F3, F4)
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "mode": self.mode, "f1": self.f1, "f2": self.f2,
                "random_kind_map": self.random_kind_map, "engine_overrides": dict(self.engine),
                "description": self.description}


ARMS: dict[str, Arm] = {
    "A0": Arm("A0", "masked", description="default masked"),
    "A1": Arm("A1", "masked", f1=True, description="masked + F1 target subsampling (t = 1e-4)"),
    "A2": Arm("A2", "masked", f1=True, engine=(("neg_budget", 1), ("cand_tries", 4)),
              description="F1 + F3 (neg_budget 1) + F4 (cand_tries 4), masked"),
    "A3": Arm("A3", "masked", f2=True, description="F2: at most M = 2,000 targets per class per epoch, masked"),
    "A4": Arm("A4", "causal", description="default causal"),
    "A5": Arm("A5", "diffusion", description="default diffusion"),
    "A6": Arm("A6", "masked", f1=True, random_kind_map=True, description="A1 with a random kind_map"),
}


def engine_config(arm: Arm | str, seed: int, vocab: int = VOCAB, n_clauses: int = N_CLAUSES, **overrides) -> EngineConfig:
    """The docs/11 EngineConfig of an arm: section 7 defaults, the arm's mode and overrides, then `overrides`
    (used only by the smoke, for example n_clauses)."""
    arm = ARMS[arm] if isinstance(arm, str) else arm
    kw: dict[str, Any] = {"mode": arm.mode, "vocab": int(vocab), "n_clauses": int(n_clauses), "seed": int(seed)}
    kw.update(dict(arm.engine))
    kw.update(overrides)
    cfg = EngineConfig(**kw)
    cfg.validate()
    return cfg


# ------------------------------------------------------------------ training prefix
def slice_prefix(corpus: CorpusArrays, n_tokens: int) -> CorpusArrays:
    """The first `n_tokens` tokens cut at a document boundary (the largest boundary <= n_tokens, at least one
    document), exactly the rule of experiments/m3_tiny_lm.py slice_train; static tables of the whole split;
    targets = every position (tests check equality with slice_train)."""
    ds = np.asarray(corpus.doc_start)
    d = int(np.searchsorted(ds, n_tokens, side="right")) - 1
    end = int(ds[max(1, d)])
    return CorpusArrays(tok=np.ascontiguousarray(corpus.tok[:end]),
                        doc_start=np.ascontiguousarray(ds[:d + 1] if ds[d] == end else ds[:d + 2]),
                        kind_map=np.ascontiguousarray(corpus.kind_map), stop=np.ascontiguousarray(corpus.stop),
                        sym_freq=np.ascontiguousarray(corpus.sym_freq), targets=np.arange(end, dtype=np.uint32))


def with_tables(corpus: CorpusArrays, targets: np.ndarray | None = None, kind_map: np.ndarray | None = None,
                sym_freq: np.ndarray | None = None) -> CorpusArrays:
    """The same token arrays with another example set and/or static tables."""
    return CorpusArrays(tok=corpus.tok, doc_start=corpus.doc_start,
                        kind_map=np.ascontiguousarray(corpus.kind_map if kind_map is None else kind_map, dtype=np.uint16),
                        stop=corpus.stop,
                        sym_freq=np.ascontiguousarray(corpus.sym_freq if sym_freq is None else sym_freq, dtype=np.uint32),
                        targets=np.ascontiguousarray(corpus.targets if targets is None else targets, dtype=np.uint32))


# ------------------------------------------------------------------ F1
def token_counts(tok: np.ndarray, vocab: int) -> np.ndarray:
    return np.bincount(np.asarray(tok, np.int64), minlength=vocab).astype(np.int64)


def f1_keep_prob(counts: np.ndarray, t: float = F1_T) -> np.ndarray:
    """float64 [V]: min(1, sqrt(t / f) + t / f) with f = counts / counts.sum(); 1 for unseen classes."""
    c = np.asarray(counts, np.float64)
    f = c / max(1.0, c.sum())
    out = np.ones_like(f)
    nz = f > 0
    r = t / f[nz]
    out[nz] = np.minimum(1.0, np.sqrt(r) + r)
    return out


def f1_keep_q32(counts: np.ndarray, t: float = F1_T) -> tuple[np.ndarray, np.ndarray]:
    """(always [V] bool, thr [V] uint64): keep p iff always[y] or (mix64(p) >> 32) < thr[y]."""
    prob = f1_keep_prob(counts, t)
    always = prob >= 1.0
    thr = np.array([q32(min(1.0, float(x))) for x in prob], dtype=np.uint64)
    return always, thr


def f1_targets(tok: np.ndarray, vocab: int, t: float = F1_T, positions: np.ndarray | None = None,
               counts: np.ndarray | None = None) -> np.ndarray:
    """u32 ascending: the positions (of `positions`, default all) kept by F1. `counts` default: the counts
    of `tok` (the training prefix)."""
    tok = np.asarray(tok)
    counts = token_counts(tok, vocab) if counts is None else np.asarray(counts, np.int64)
    always, thr = f1_keep_q32(counts, t)
    p = np.arange(tok.shape[0], dtype=np.int64) if positions is None else np.asarray(positions, np.int64)
    y = tok[p].astype(np.int64)
    r32 = mix64_np(p.astype(np.uint64)) >> np.uint64(32)
    keep = always[y] | (r32 < thr[y])
    return p[keep].astype(np.uint32)


# ------------------------------------------------------------------ F2
def f2_targets(tok: np.ndarray, epoch: int, m: int = F2_M, positions: np.ndarray | None = None) -> np.ndarray:
    """u32 ascending: per class at most `m` of the candidate positions (default all), the ones with the
    smallest rng64(0, STREAM_F2, epoch, p, 0) (ties toward the lower position) when the class has more."""
    tok = np.asarray(tok)
    p = np.arange(tok.shape[0], dtype=np.int64) if positions is None else np.sort(np.asarray(positions, np.int64))
    if p.size == 0:
        return p.astype(np.uint32)
    y = tok[p].astype(np.int64)
    key = rng64_np(0, STREAM_F2, np.uint64(int(epoch)), p.astype(np.uint64), np.uint64(0))
    order = np.lexsort((p, key, y))                  # class, then key, then position
    ys = y[order]
    first = np.ones(ys.shape[0], bool)
    first[1:] = ys[1:] != ys[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(ys.shape[0]), 0))
    rank = np.arange(ys.shape[0]) - start
    return np.sort(p[order[rank < int(m)]]).astype(np.uint32)


# ------------------------------------------------------------------ random kind_map (A6)
def random_permutation(vocab: int) -> np.ndarray:
    """int64 [V]: pi, the token ids ordered by rng64(0, STREAM_KM, t, 0, 0) (stable, ties toward lower t)."""
    key = rng64_np(0, STREAM_KM, np.arange(vocab, dtype=np.uint64), np.uint64(0), np.uint64(0))
    return np.argsort(key, kind="stable").astype(np.int64)


def sym_freq_from_counts(kind_map: np.ndarray, counts: np.ndarray, vocab: int, n_pol: int = 0) -> np.ndarray:
    """docs/11 sym_freq for a kind_map from per-token counts (tmlm.data.synthetic.sym_freq_of without the
    token pass): freq[kind_map[k][t]] += counts[t] for k = 0..3, capped at 2^32 - 1; MASK and EDGE max."""
    freq = np.zeros(n_sym(vocab, n_pol), dtype=np.uint64)
    c = np.asarray(counts, np.uint64)
    for k in range(4):
        np.add.at(freq, np.asarray(kind_map[k], np.int64), c)
    freq = np.minimum(freq, np.uint64(0xFFFFFFFF)).astype(np.uint32)
    freq[sym_mask(vocab)] = 0xFFFFFFFF
    freq[sym_edge(vocab)] = 0xFFFFFFFF
    return freq


def random_kind_map(kind_map: np.ndarray, sym_freq: np.ndarray, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """(kind_map', sym_freq') of A6. The token counts are read from sym_freq[:V] (kind 0 is the identity, so
    sym_freq[t] is the training count of token t, docs/11 section 2)."""
    km = np.asarray(kind_map, np.uint16)
    pi = random_permutation(vocab)
    out = km.copy()
    for k in (1, 2, 3):
        out[k] = km[k][pi]
    counts = np.asarray(sym_freq[:vocab], np.uint64)
    return out, sym_freq_from_counts(out, counts, vocab)


# ------------------------------------------------------------------ per-arm corpora
@dataclass
class ArmData:
    """The arm's view of the training prefix: tables and the example set per epoch."""

    arm: Arm
    prefix: CorpusArrays
    kind_map: np.ndarray
    sym_freq: np.ndarray
    f1_positions: np.ndarray | None
    info: dict

    def targets(self, epoch: int) -> np.ndarray:
        base = self.f1_positions
        if self.arm.f2:
            return f2_targets(self.prefix.tok, epoch, F2_M, positions=base)
        if base is not None:
            return base
        return np.arange(self.prefix.tok.shape[0], dtype=np.uint32)

    def corpus(self, epoch: int) -> CorpusArrays:
        return with_tables(self.prefix, targets=self.targets(epoch), kind_map=self.kind_map, sym_freq=self.sym_freq)

    @property
    def targets_vary_by_epoch(self) -> bool:
        return bool(self.arm.f2)

    def feature_tables(self) -> CorpusArrays:
        """A reference object with the arm's kind_map, stop and sym_freq (for tmlm.ssl.features.corpus_arrays)."""
        return with_tables(self.prefix, kind_map=self.kind_map, sym_freq=self.sym_freq)


def arm_data(arm: Arm | str, prefix: CorpusArrays, vocab: int = VOCAB, t: float = F1_T) -> ArmData:
    arm = ARMS[arm] if isinstance(arm, str) else arm
    km, sf = np.asarray(prefix.kind_map), np.asarray(prefix.sym_freq)
    info: dict[str, Any] = {"prefix_tokens": int(prefix.tok.shape[0]), "prefix_docs": int(prefix.doc_start.shape[0] - 1)}
    if arm.random_kind_map:
        km, sf = random_kind_map(km, sf, vocab)
        info["kind_map_changed_fraction"] = {f"kind{k}": float(np.mean(km[k] != np.asarray(prefix.kind_map)[k])) for k in (1, 2, 3)}
    f1p = None
    if arm.f1:
        f1p = f1_targets(prefix.tok, vocab, t)
        info["f1_kept"] = int(f1p.shape[0])
        info["f1_kept_fraction"] = float(f1p.shape[0] / max(1, prefix.tok.shape[0]))
        info["f1_t"] = float(t)
    return ArmData(arm=arm, prefix=prefix, kind_map=np.ascontiguousarray(km, dtype=np.uint16),
                   sym_freq=np.ascontiguousarray(sf, dtype=np.uint32), f1_positions=f1p, info=info)


def expected_f1_fraction(counts: np.ndarray, t: float = F1_T) -> float:
    """sum_y count(y) * keep_prob(y) / N (derived, for logs)."""
    c = np.asarray(counts, np.float64)
    return float((c * f1_keep_prob(c, t)).sum() / max(1.0, c.sum()))


__all__ = ["ARMS", "Arm", "ArmData", "arm_data", "engine_config", "slice_prefix", "with_tables", "f1_targets",
           "f1_keep_prob", "f2_targets", "random_kind_map", "sym_freq_from_counts", "mix64_np", "rng64_np",
           "expected_f1_fraction"]
