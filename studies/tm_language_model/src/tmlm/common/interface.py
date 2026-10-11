"""The one model interface shared by all three tracks (design decision).

Track A (GraphTM), Track B (sparse event-driven engine) and every Track C baseline
implement `MaskedTokenModel`, so that evaluation, calibration, decoding, certification
and the leakage tests are written once. This file defines the contract only. It holds
no model code.

Conventions used below:
    N_tok  number of tokens in a stream        V  vocabulary size (number of classes)
    N_doc  number of documents in a stream     B  number of target positions in a batch
    M      number of masked positions

The design writes `explain(position)`. A position only means something together with its
token stream and its mask set, so the method takes `(batch, index)`, where `index` is a
row of the batch and the position is `batch.positions[index]`.

THE LEAKAGE RULE. A model must never read `batch.tokens` at any index listed in
`batch.masked_positions`. Those tokens are the answers. The metamorphic test of the
design replaces them with random ids and demands bit-identical scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class MaskedBatch:
    """Target positions inside one token stream. Arrays are views, never copies.

    Attributes:
        tokens: uint16 [N_tok]. Token ids of one split, all documents concatenated.
            Every id is < V. Read-only by convention.
        doc_start: int64 [N_doc + 1]. Strictly increasing, doc_start[0] == 0 and
            doc_start[-1] == N_tok. Document d is tokens[doc_start[d]:doc_start[d + 1]].
            Context never crosses a document boundary.
        masked_positions: int64 [M]. Strictly increasing indices into `tokens`. Every
            listed position is hidden from every context (it reads as MASK), for all
            rows of the batch, not only for its own row.
        positions: int64 [B]. The positions to predict, in any order, duplicates
            allowed. In masked and diffusion configs each must be listed in
            `masked_positions`. The gold token of row i is tokens[positions[i]]: it is
            for the evaluator, never for the model.
        causal: False for masked and diffusion configs. True for the causal config:
            row i may only use tokens at indices < positions[i] inside its document,
            and `masked_positions` must then be empty.
    """

    tokens: np.ndarray
    doc_start: np.ndarray
    masked_positions: np.ndarray
    positions: np.ndarray
    causal: bool = False

    def validate(self, vocab_size: int | None = None) -> None:
        """Raise ValueError when dtypes, shapes or invariants above are violated.

        Cost is O(N_doc + M + B), plus O(N_tok) when `vocab_size` is given.
        """
        for name, dtype in (
            ("tokens", np.uint16),
            ("doc_start", np.int64),
            ("masked_positions", np.int64),
            ("positions", np.int64),
        ):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.dtype != dtype or arr.ndim != 1:
                raise ValueError(f"{name} must be a 1-d numpy array of dtype {np.dtype(dtype)}")
        n_tok = self.tokens.shape[0]
        if self.doc_start.size < 2 or self.doc_start[0] != 0 or self.doc_start[-1] != n_tok:
            raise ValueError("doc_start must start at 0 and end at N_tok")
        if np.any(np.diff(self.doc_start) <= 0):
            raise ValueError("doc_start must be strictly increasing")
        if self.masked_positions.size:
            if self.masked_positions[0] < 0 or self.masked_positions[-1] >= n_tok:
                raise ValueError("masked_positions out of range")
            if np.any(np.diff(self.masked_positions) <= 0):
                raise ValueError("masked_positions must be strictly increasing")
        if self.positions.size and (self.positions.min() < 0 or self.positions.max() >= n_tok):
            raise ValueError("positions out of range")
        if self.causal:
            if self.masked_positions.size:
                raise ValueError("causal batches must have empty masked_positions")
        elif not np.all(np.isin(self.positions, self.masked_positions)):
            raise ValueError("every target position must be listed in masked_positions")
        if vocab_size is not None and n_tok and int(self.tokens.max()) >= vocab_size:
            raise ValueError("token id >= vocab_size")


@dataclass(frozen=True)
class FiredClause:
    """One clause that fired for one target position.

    Attributes:
        clause_id: index of the clause in the exported bank (`export_bank`).
        literals: the INCLUDED literals of the clause, as ids in the model's literal
            space (uint32 range). Track B: `(segment << 17) | (symbol << 1) | negated`.
        literal_text: one human-readable string per entry of `literals`, same order,
            for example "t[-1]=New" or "NOT t[+2]=the".
        votes: (class id, signed integer weight) pairs this clause adds to the class
            sums. Sparse: classes that are not listed get 0 from this clause.
    """

    clause_id: int
    literals: tuple[int, ...]
    literal_text: tuple[str, ...]
    votes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Explanation:
    """Everything needed to re-derive one prediction by hand, and to build a receipt.

    Attributes:
        position: the explained index into `tokens`.
        predicted: argmax class id, ties broken toward the lowest class id.
        score: the raw score of `predicted`, equal to predict_scores(...)[index, predicted].
        clauses: ALL clauses that fired for this position, ordered by clause_id. The
            sum of their votes per class must reproduce the row of `predict_scores`
            exactly for clause-based models. Baselines that have no clauses return ().
    """

    position: int
    predicted: int
    score: float
    clauses: tuple[FiredClause, ...]


@runtime_checkable
class MaskedTokenModel(Protocol):
    """What every model and every baseline must provide.

    All methods are pure with respect to learnable state: calling them never changes
    `state_hash()`. All are deterministic: same state and same batch give bit-identical
    outputs, independent of batch composition (row i does not depend on the other
    rows), thread count, or GPU launch configuration.
    """

    vocab_size: int

    def predict_scores(self, batch: MaskedBatch) -> np.ndarray:
        """Raw, uncalibrated class scores.

        Returns:
            [B, V] array, C-contiguous, on the host. dtype int32 for clause-based
            models (the exact integer class sums), float32 for baselines. Higher is
            better. Row i belongs to batch.positions[i]. The evaluator takes top-k with
            ties broken toward the lowest class id.
        """
        ...

    def logprobs(self, batch: MaskedBatch) -> np.ndarray:
        """Calibrated natural-log probabilities over the vocabulary.

        Returns:
            [B, V] float32, C-contiguous, on the host. Every row is a normalized
            distribution: logsumexp(row) is 0 within 1e-4. No entry is -inf (a model
            must smooth), so perplexity is always finite. Calibration parameters are
            fitted on validation data only and are part of `state_hash()`.
        """
        ...

    def explain(self, batch: MaskedBatch, index: int) -> Explanation:
        """The fired clauses, with literals and votes, behind row `index` of `batch`.

        Must agree with `predict_scores(batch)[index]` as described in `Explanation`.
        """
        ...

    def state_hash(self) -> str:
        """sha256 hex digest (64 chars) of ALL learnable state in a canonical order.

        Covers clause literals and automaton states, votes, biases, calibration
        parameters, and any RNG counters that influence future training. Must be equal
        across implementations of the same spec (oracle, CPU, GPU) after the same
        steps, and must not depend on memory layout, padding, or device. Used by the
        replay and equivalence tests.
        """
        ...

    def export_bank(self) -> dict[str, Any]:
        """A self-contained dump that a standalone receipt checker can load.

        The checker imports nothing from this project, so the dump uses only builtin
        scalars, str, and numpy arrays, and must survive `np.savez` and `np.load`
        without pickling. Required keys:
            "format"       str, names the track and literal encoding, e.g. "tmlm.bank.trackB.v1"
            "vocab_size"   int, V
            "state_hash"   str, `state_hash()` at export time
            "lit_ptr"      int64 [C + 1], CSR row pointers into `lit_id`
            "lit_id"       uint32 [nnz_lit], included literal ids of clause c at
                           lit_id[lit_ptr[c]:lit_ptr[c + 1]]
            "vote_ptr"     int64 [C + 1], CSR row pointers into the vote arrays
            "vote_cls"     int32 [nnz_vote], class ids
            "vote_w"       int32 [nnz_vote], signed weights
        Track specific keys (literal budget, class bias, calibration scalars) are
        defined by the track's own spec. Baselines without clauses raise
        NotImplementedError.
        """
        ...
