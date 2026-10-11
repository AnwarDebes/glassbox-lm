"""The engine API every implementation of docs/11-engine-spec.md must provide.

Three implementations (Python oracle, C++ host engine, CUDA engine) subclass `EngineBase`.
They must agree BIT FOR BIT on `state_hash()` and on every counter in `DIAG_COUNTERS` after any
sequence of calls. This module holds the shared data containers, the canonical state layout
and the canonical hash; it holds no learning logic.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import CLS_EMPTY, LIT_EMPTY, Derived, EngineConfig, derive, n_sym


@dataclass(frozen=True)
class CorpusArrays:
    """Language input (spec section 1.1). Arrays are read-only views.

    tok        u16 [N_tok]      token ids < V
    doc_start  u32 [N_doc + 1]  strictly increasing, first 0, last N_tok
    kind_map   u16 [4, V]       kind_map[0][t] == t; kind_map[k][t] = kind-k cluster symbol
    stop       u8  [V]          1 for the N_STOP most frequent training tokens
    sym_freq   u32 [N_SYM]      training frequency of every symbol (anchor choice); SYM_MASK and
                                SYM_EDGE hold 0xFFFFFFFF
    targets    u32 [N_ex]       positions that are examples; the epoch permutation runs over
                                indices into this array
    """

    tok: np.ndarray
    doc_start: np.ndarray
    kind_map: np.ndarray
    stop: np.ndarray
    sym_freq: np.ndarray
    targets: np.ndarray

    def validate(self, cfg: EngineConfig) -> None:
        v = cfg.vocab
        checks = (
            ("tok", self.tok, np.uint16, 1),
            ("doc_start", self.doc_start, np.uint32, 1),
            ("kind_map", self.kind_map, np.uint16, 2),
            ("stop", self.stop, np.uint8, 1),
            ("sym_freq", self.sym_freq, np.uint32, 1),
            ("targets", self.targets, np.uint32, 1),
        )
        for name, arr, dtype, ndim in checks:
            if not isinstance(arr, np.ndarray) or arr.dtype != dtype or arr.ndim != ndim:
                raise ValueError(f"{name}: expected {np.dtype(dtype)} with {ndim} dims")
        if self.kind_map.shape != (4, v) or self.stop.shape != (v,):
            raise ValueError("kind_map must be [4, V] and stop [V]")
        if self.sym_freq.shape != (n_sym(v, cfg.n_pol),):
            raise ValueError("sym_freq must be [N_SYM]")
        if self.doc_start[0] != 0 or self.doc_start[-1] != self.tok.shape[0]:
            raise ValueError("doc_start must start at 0 and end at N_tok")
        if self.tok.size and int(self.tok.max()) >= v:
            raise ValueError("token id >= V")
        if not np.array_equal(self.kind_map[0], np.arange(v, dtype=np.uint16)):
            raise ValueError("kind_map[0] must be the identity")


@dataclass(frozen=True)
class TabularData:
    """Tabular input (spec section 6). x: u8 [N, F] in {0, 1}; y: u16 [N] class ids < cfg.vocab."""

    x: np.ndarray
    y: np.ndarray

    def validate(self, cfg: EngineConfig) -> None:
        if self.x.dtype != np.uint8 or self.x.ndim != 2 or self.x.shape[1] != cfg.n_features:
            raise ValueError("x must be u8 [N, F]")
        if self.y.dtype != np.uint16 or self.y.shape != (self.x.shape[0],):
            raise ValueError("y must be u16 [N]")
        if self.y.size and int(self.y.max()) >= cfg.vocab:
            raise ValueError("class id >= number of classes")


# Canonical state layout and hash order (spec section 2). (name, dtype, shape-kind)
STATE_FIELDS: tuple[tuple[str, type, str], ...] = (
    ("lit_id", np.uint32, "C,L"),
    ("lit_state", np.uint8, "C,L"),
    ("vote_cls", np.uint16, "C,V"),
    ("vote_w", np.int16, "C,V"),
    ("decay_mark", np.uint64, "C"),
    ("own_fire_mass", np.uint64, "C"),
    ("stat_fire", np.uint32, "C"),
    ("stat_tp", np.uint32, "C"),
    ("birth_step", np.uint32, "C"),
    ("last_tp_step", np.uint32, "C"),
    ("alive", np.uint8, "C"),
    ("G", np.uint64, "K"),
    ("step", np.uint64, "1"),
)

# Integer diagnostics returned by train calls. All implementations must report identical values.
DIAG_COUNTERS: tuple[str, ...] = (
    "n_examples",
    "n_candidates",
    "n_fired",
    "n_events_applied",
    "n_events_skipped_not_firing",
    "n_events_dropped_cap",
    "n_tgt_gate_pass",
    "n_neg_gate_pass",
    "n_type1",
    "n_type2",
    "n_inserted",
    "n_neg_inserted",
    "n_vetoes",
    "n_seed_requests",
    "n_seeded",
    "n_recycled",
)


def state_shapes(cfg: EngineConfig) -> dict[str, tuple[int, ...]]:
    dims = {"C,L": (cfg.n_clauses, cfg.l_slots), "C,V": (cfg.n_clauses, cfg.v_slots),
            "C": (cfg.n_clauses,), "K": (cfg.vocab,), "1": (1,)}
    return {name: dims[kind] for name, _, kind in STATE_FIELDS}


def empty_state(cfg: EngineConfig) -> dict[str, np.ndarray]:
    """The initial state of a sparse engine: every clause free, every slot cleared (spec section 2)."""
    shapes = state_shapes(cfg)
    st = {name: np.zeros(shapes[name], dtype=dtype) for name, dtype, _ in STATE_FIELDS}
    st["lit_id"][:] = LIT_EMPTY
    st["vote_cls"][:] = CLS_EMPTY
    return st


def state_hash_of(state: dict[str, np.ndarray], cfg: EngineConfig) -> str:
    """SHA-256 over the little-endian bytes of the state arrays in canonical order."""
    shapes = state_shapes(cfg)
    h = hashlib.sha256()
    for name, dtype, _ in STATE_FIELDS:
        arr = np.ascontiguousarray(state[name])
        if arr.dtype != np.dtype(dtype) or arr.shape != shapes[name]:
            raise ValueError(f"state[{name!r}]: expected {np.dtype(dtype)} {shapes[name]}, "
                             f"got {arr.dtype} {arr.shape}")
        h.update(arr.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


class EngineBase(ABC):
    """Common API. `epoch` and positions fully determine all randomness (spec section 3)."""

    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays | None = None,
                 tabular: TabularData | None = None) -> None:
        cfg.validate()
        if (cfg.mode == "tabular") != (tabular is not None) or (cfg.mode != "tabular") != (corpus is not None):
            raise ValueError("tabular mode needs `tabular`, every other mode needs `corpus`")
        (tabular if tabular is not None else corpus).validate(cfg)  # type: ignore[union-attr]
        self.cfg = cfg
        self.derived: Derived = derive(cfg)
        self.corpus = corpus
        self.tabular = tabular

    @property
    def n_examples(self) -> int:
        if self.tabular is not None:
            return int(self.tabular.y.shape[0])
        return int(self.corpus.targets.shape[0])  # type: ignore[union-attr]

    def n_batches(self) -> int:
        return (self.n_examples + self.cfg.batch - 1) // self.cfg.batch

    @abstractmethod
    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        """Run batch `batch_index` of epoch `epoch` exactly as spec section 5. Returns DIAG_COUNTERS."""

    @abstractmethod
    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        """Treat `examples` (u64 positions, or row indices in tabular mode) as ONE batch in the given
        order. Same semantics as train_batch, used by tests. Increments `step` by one."""

    @abstractmethod
    def predict_scores(self, epoch: int, examples: np.ndarray,
                       hidden: np.ndarray | None = None) -> np.ndarray:
        """i32 [n, V] exact class sums (spec section 4, steps 1 to 4). Never changes the state.
        `hidden`: optional u8 [N_tok] bitmap that REPLACES the hidden() function (frozen evaluation
        sets); ignored in causal and tabular modes. The target position is always hidden."""

    @abstractmethod
    def fired_clauses(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        """u32 ascending ids of the clauses that fire for one example. Never changes the state."""

    @abstractmethod
    def state(self) -> dict[str, np.ndarray]:
        """Copies of all arrays of STATE_FIELDS, host numpy, canonical dtypes and shapes."""

    @abstractmethod
    def load_state(self, state: dict[str, np.ndarray]) -> None:
        """Replace the full state (derived structures are rebuilt)."""

    def state_hash(self) -> str:
        return state_hash_of(self.state(), self.cfg)
