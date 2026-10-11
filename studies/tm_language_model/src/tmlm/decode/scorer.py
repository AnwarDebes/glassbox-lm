"""A trained Track B model bound to a generation window (docs/03 "Track C: decoders"; docs/04 M8).

`TrainedModel` holds everything a decoder needs from a training run: the engine config, the
static tables of the training corpus (kind_map, stop, sym_freq), the canonical state, the
integer class bias (train unigram) and the calibration scalars. `bind(length)` returns a
`WindowModel`: a CpuEngine whose corpus is ONE document of `length` positions whose tokens and
hidden flags the decoders mutate in place (the engine reads the very same buffers; bag
occurrence lists are refreshed after every change through `CpuEngine.refresh_occurrences`).
Scores come from the C engine's forward pass (docs/11 section 4, steps 1 to 4), so a decoder
built on `WindowModel` is scored by the reference implementation, and `IncrementalEvaluator`
(incremental.py) is checked against it.

Nothing here draws random numbers; the decoders do, through the counter-based RNG only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..engine.api import CorpusArrays, state_hash_of
from ..engine.config import EngineConfig
from ..engine.cpu.cpu_engine import CpuEngine
from .bank import Bank, bank_sha256, class_bias_from_counts
from .conftm import Calibration


@dataclass(frozen=True)
class StaticTables:
    """The vocabulary-level tables of the training corpus (docs/11 section 1)."""

    kind_map: np.ndarray  # u16 [4, V]
    stop: np.ndarray  # u8 [V]
    sym_freq: np.ndarray  # u32 [N_SYM]

    @staticmethod
    def of(corpus: CorpusArrays) -> "StaticTables":
        return StaticTables(kind_map=np.ascontiguousarray(corpus.kind_map, dtype=np.uint16),
                            stop=np.ascontiguousarray(corpus.stop, dtype=np.uint8),
                            sym_freq=np.ascontiguousarray(corpus.sym_freq, dtype=np.uint32))


class TrainedModel:
    """Engine config + static tables + state + class bias + calibration."""

    def __init__(self, cfg: EngineConfig, tables: StaticTables, state: dict[str, np.ndarray],
                 class_bias: np.ndarray, calibration: Calibration | None = None) -> None:
        cfg.validate()
        if cfg.mode == "tabular":
            raise ValueError("decoders need a language mode")
        self.cfg = cfg
        self.tables = tables
        self.state = {k: np.ascontiguousarray(v) for k, v in state.items()}
        self.class_bias = np.ascontiguousarray(class_bias, dtype=np.int32)
        self.calibration = calibration if calibration is not None else Calibration.identity()
        self.state_hash = state_hash_of(self.state, cfg)
        self._bank: Bank | None = None

    @staticmethod
    def from_engine(engine, train_tok: np.ndarray, calibration: Calibration | None = None) -> "TrainedModel":
        counts = np.bincount(np.asarray(train_tok, dtype=np.int64), minlength=engine.cfg.vocab)
        return TrainedModel(engine.cfg, StaticTables.of(engine.corpus), engine.state(),
                            class_bias_from_counts(counts, engine.cfg.vocab), calibration)

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab

    def bank(self) -> Bank:
        if self._bank is None:
            self._bank = Bank.from_state(self.state, self.cfg, self.tables.kind_map, self.tables.stop,
                                         self.tables.sym_freq, self.class_bias, self.calibration.a_q16,
                                         self.calibration.b_q16, self.state_hash)
        return self._bank

    def export_bank(self) -> dict[str, Any]:
        return self.bank().export()

    def bank_sha256(self) -> str:
        return bank_sha256(self.export_bank())

    def with_calibration(self, calibration: Calibration) -> "TrainedModel":
        return TrainedModel(self.cfg, self.tables, self.state, self.class_bias, calibration)

    def bind(self, length: int) -> "WindowModel":
        return WindowModel(self, length)

    def save(self, path) -> None:
        np.savez(path, cfg=np.array(repr(self.cfg.to_dict())), class_bias=self.class_bias,
                 a_q16=np.array(self.calibration.a_q16), b_q16=np.array(self.calibration.b_q16),
                 kind_map=self.tables.kind_map, stop=self.tables.stop, sym_freq=self.tables.sym_freq,
                 **{f"state_{k}": v for k, v in self.state.items()})

    @staticmethod
    def load(path) -> "TrainedModel":
        import ast
        with np.load(path, allow_pickle=False) as z:
            cfg = EngineConfig(**ast.literal_eval(str(z["cfg"])))
            tables = StaticTables(kind_map=z["kind_map"], stop=z["stop"], sym_freq=z["sym_freq"])
            state = {k[len("state_"):]: z[k] for k in z.files if k.startswith("state_")}
            cal = Calibration(int(z["a_q16"]), int(z["b_q16"]))
            return TrainedModel(cfg, tables, state, z["class_bias"], cal)


class WindowModel:
    """One document window of `length` positions, scored by the C engine.

    `tokens` (u16 [L]) and `hidden` (u8 [L]) are the engine's own buffers. A hidden position
    reads as MASK whatever its token value. Use `set_token`, `clear` and `set_window` to change
    them; never write the arrays directly, the bag occurrence lists would go stale.
    """

    def __init__(self, model: TrainedModel, length: int) -> None:
        if length < 1:
            raise ValueError("length must be >= 1")
        self.model = model
        self.cfg = model.cfg
        tok = np.zeros(length, dtype=np.uint16)
        corpus = CorpusArrays(tok=tok, doc_start=np.array([0, length], dtype=np.uint32),
                              kind_map=model.tables.kind_map, stop=model.tables.stop,
                              sym_freq=model.tables.sym_freq, targets=np.arange(length, dtype=np.uint32))
        self.engine = CpuEngine(self.cfg, corpus=corpus)
        self.engine.load_state(model.state)
        self.tokens: np.ndarray = self.engine.corpus_buffers["tok"]
        self.hidden = np.ones(length, dtype=np.uint8)
        self.L = length
        self._bags = self.cfg.n_bag_k > 0
        self.n_engine_calls = 0

    # ---------------------------------------------------------------- window mutation
    def set_token(self, q: int, t: int) -> None:
        self.tokens[q] = t
        self.hidden[q] = 0
        if self._bags:
            self.engine.refresh_occurrences()

    def clear(self, q: int) -> None:
        self.hidden[q] = 1
        if self._bags:
            self.engine.refresh_occurrences()

    def set_window(self, tokens: np.ndarray, hidden: np.ndarray) -> None:
        tokens, hidden = np.asarray(tokens), np.asarray(hidden)
        if tokens.shape != (self.L,) or hidden.shape != (self.L,):
            raise ValueError("window arrays must have the bound length")
        self.tokens[:] = tokens.astype(np.uint16)
        self.hidden[:] = (hidden != 0).astype(np.uint8)
        if self._bags:
            self.engine.refresh_occurrences()

    def open_positions(self) -> np.ndarray:
        return np.flatnonzero(self.hidden).astype(np.int64)

    # ---------------------------------------------------------------- scoring (state never changes)
    def _bitmap(self) -> np.ndarray | None:
        return self.hidden if self.cfg.mode in ("masked", "diffusion") else None

    def scores(self, positions: np.ndarray) -> np.ndarray:
        """int32 [n, V] exact class sums for the given window positions (each read as MASK)."""
        self.n_engine_calls += 1
        return self.engine.predict_scores(0, np.asarray(positions, dtype=np.uint32), self._bitmap())

    def fired(self, p: int) -> np.ndarray:
        """u32 ascending ids of the clauses that fire for window position p."""
        return self.engine.fired_clauses(0, int(p), self._bitmap())

    def logprobs(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return self.model.calibration.logprobs(scores, self.model.class_bias, temperature)
