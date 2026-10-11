"""Incremental re-evaluation for generation (docs/03 "Diffusion sampler spec": consecutive partial
inputs are similar; docs/04 T2.3; docs/13 H6.4).

`IncrementalEvaluator` keeps, for every hidden (open) position p of a window, the candidate
clauses reached through the anchor inverted index and the subset that fires. When position q
changes (a token is written or the position is re-hidden), only the clauses whose literals can
change truth are re-verified: for an open position p at offset o = q - p, the candidates with a
literal in exact segment seg(o) when |o| <= 8, and those with a bag literal in the bucket that
contains |o| on q's side; plus the clauses newly reachable through the posting lists of the new
symbols at (seg(o), kind k). Every other clause keeps its truth value, because a literal's truth
depends only on the symbols of the positions it reads (docs/11 section 1.4). Clause truth itself
is computed by `WindowEval` (bank.py) with the exact literal semantics. The result must be
IDENTICAL to a full re-evaluation by the C engine; `tests/test_decode.py` asserts it after every
change on random windows.

The evaluator offers the same scoring API as `WindowModel` (scorer.py), so the decoders run on
either; the C engine's full re-evaluation is the reference and stays the reference (H6.4).
"""

from __future__ import annotations

import numpy as np

from ..engine.config import N_EXACT_SEG, bag_seg, lit_pack, offset_seg, seg_offset, sym_edge, sym_mask
from .bank import WindowEval


class IncrementalEvaluator:
    def __init__(self, model, length: int) -> None:
        if length < 1:
            raise ValueError("length must be >= 1")
        self.model = model
        self.cfg = model.cfg
        self.bank = model.bank()
        self.L = int(length)
        self.tokens = np.zeros(self.L, dtype=np.uint16)
        self.hidden = np.ones(self.L, dtype=np.uint8)
        self.we = WindowEval(self.bank, self.tokens, self.hidden)  # shares the two arrays
        self.posting = self.bank.posting_lists()
        self.smask, self.sedge = sym_mask(self.bank.vocab), sym_edge(self.bank.vocab)
        self.enabled = self.we.enabled
        self.bag_reach = 8 * (1 << self.cfg.n_bag_k) if self.cfg.n_bag_k > 0 else 0
        # per clause: the distinct segments its literals read
        b = self.bank
        self.clause_segs: list[np.ndarray] = [np.unique(b.lit_id[b.lit_ptr[j]:b.lit_ptr[j + 1]] >> 17) for j in range(b.n_clauses)]
        self.cand: dict[int, set[int]] = {}
        self.fired_sets: dict[int, set[int]] = {}
        self.touch: dict[int, dict[int, set[int]]] = {}
        self.n_verified = 0
        self.n_engine_calls = 0
        for p in range(self.L):
            self._open(p)

    # ---------------------------------------------------------------- bookkeeping
    def _lookup_lits(self, p: int) -> list[int]:
        """The anchor literals looked up for target p (docs/11 section 4 step 2)."""
        lits = []
        for seg in range(N_EXACT_SEG):
            if not self.enabled[seg]:
                continue
            q = p + seg_offset(seg)
            if q < 0 or q >= self.L:
                lits.append(lit_pack(seg, self.sedge, 0))
            elif q == p or self.we.is_hidden(q):
                lits.append(lit_pack(seg, self.smask, 0))
            else:
                lits.extend(lit_pack(seg, int(self.bank.kind_map[k, self.tokens[q]]), 0) for k in range(4))
        return lits

    def _add_candidates(self, p: int, lits: list[int]) -> set[int]:
        new: set[int] = set()
        cand, touch = self.cand[p], self.touch[p]
        for lit in lits:
            post = self.posting.get(lit)
            if post is None:
                continue
            for j in post.tolist():
                if j not in cand:
                    cand.add(j)
                    new.add(j)
                    for seg in self.clause_segs[j].tolist():
                        touch.setdefault(int(seg), set()).add(j)
        return new

    def _verify(self, p: int, ids: set[int]) -> None:
        if not ids:
            return
        arr = np.fromiter(ids, dtype=np.int64, count=len(ids))
        arr.sort()
        fires = self.we.fires(p, arr)
        self.n_verified += int(arr.shape[0])
        fs = self.fired_sets[p]
        for j, f in zip(arr.tolist(), fires.tolist()):
            if f:
                fs.add(j)
            else:
                fs.discard(j)

    def _open(self, p: int) -> None:
        self.cand[p], self.fired_sets[p], self.touch[p] = set(), set(), {}
        new = self._add_candidates(p, self._lookup_lits(p))
        self._verify(p, new)

    def _changed(self, q: int) -> None:
        """Position q changed symbol (token written or re-hidden): update every open p in reach."""
        for p in list(self.cand):
            if p == q:
                continue
            o = q - p
            d = abs(o)
            if d <= 8:
                seg = offset_seg(o)
                if not self.enabled[seg]:
                    continue
                if self.we.is_hidden(q):
                    lits = [lit_pack(seg, self.smask, 0)]
                else:
                    lits = [lit_pack(seg, int(self.bank.kind_map[k, self.tokens[q]]), 0) for k in range(4)]
                ids = set(self.touch[p].get(seg, ()))
                ids |= self._add_candidates(p, lits)
                self._verify(p, ids)
            elif d <= self.bag_reach:
                k = 0  # the bucket with 8 * 2^k < d <= 8 * 2^(k + 1)
                while 8 * (1 << (k + 1)) < d:
                    k += 1
                seg = bag_seg(k, 0 if q < p else 1)
                if seg < self.enabled.shape[0] and self.enabled[seg]:
                    self._verify(p, set(self.touch[p].get(seg, ())))

    # ---------------------------------------------------------------- the scorer API
    def set_token(self, q: int, t: int) -> None:
        self.we.set_token(q, t)
        if q in self.cand:
            del self.cand[q], self.fired_sets[q], self.touch[q]
        self._changed(q)

    def clear(self, q: int) -> None:
        self.we.clear(q)
        self._changed(q)
        if q not in self.cand:
            self._open(q)

    def set_window(self, tokens: np.ndarray, hidden: np.ndarray) -> None:
        self.tokens[:] = np.asarray(tokens, dtype=np.uint16)
        self.hidden[:] = (np.asarray(hidden) != 0).astype(np.uint8)
        self.we._bag_cache.clear()
        self.cand, self.fired_sets, self.touch = {}, {}, {}
        for p in np.flatnonzero(self.hidden):
            self._open(int(p))

    def open_positions(self) -> np.ndarray:
        return np.flatnonzero(self.hidden).astype(np.int64)

    def _evaluate_once(self, p: int) -> np.ndarray:
        """Fired clauses of a VISIBLE position read as the target (its own token hidden), computed
        from the posting lists without keeping state (the remasking step of the diffusion sampler
        scores visible positions; the engine does the same through predict_scores)."""
        ids: set[int] = set()
        for lit in self._lookup_lits(p):
            post = self.posting.get(lit)
            if post is not None:
                ids.update(post.tolist())
        if not ids:
            return np.zeros(0, dtype=np.uint32)
        arr = np.fromiter(ids, dtype=np.int64, count=len(ids))
        arr.sort()
        fires = self.we.fires(p, arr)
        self.n_verified += int(arr.shape[0])
        return arr[fires].astype(np.uint32)

    def fired(self, p: int) -> np.ndarray:
        p = int(p)
        if p not in self.fired_sets:
            return self._evaluate_once(p)
        return np.array(sorted(self.fired_sets[p]), dtype=np.uint32)

    def scores(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        out = np.zeros((positions.shape[0], self.bank.vocab), dtype=np.int32)
        for i, p in enumerate(positions):
            out[i] = self.bank.sums(self.fired(int(p)))
        return out

    def logprobs(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return self.model.calibration.logprobs(scores, self.model.class_bias, temperature)
