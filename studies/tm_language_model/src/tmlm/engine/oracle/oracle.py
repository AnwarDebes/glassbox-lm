"""The Python ORACLE: the executable form of docs/11-engine-spec.md.

Clarity first. Every learning decision is made with plain Python integers; numpy appears only
at the API boundary (corpus arrays in, state arrays out). Functions are named after the spec
sections they implement, and the comments cite the section numbers.

Internal storage: the state arrays of spec section 2 are held as Python lists (`lit_id[j][k]`,
`vote_w[j][k]`, `G[c]`, ...). `state()` converts them to the canonical numpy layout of
`api.STATE_FIELDS`; numpy refuses out-of-range integers there, which doubles as a range check.

Readings chosen where the spec is silent or ambiguous (reported as SPEC NOTES, see the block
at the end of this docstring). Each one is implemented in exactly one clearly marked place.

N1  (5, 5.7) The sweep runs at the end of a batch when `(step + 1) % SWEEP_PERIOD == 0`, that
    is after batches number 256, 512, ... (one-based). `step` inside the sweep is the value
    BEFORE the increment. Ages use 32-bit wrapping subtraction (birth_step and last_tp_step
    are truncated to 32 bits).
N2  (5.1) E_CAP limits the events CONSIDERED: the event list of a clause is cut to its first
    E_CAP entries in ascending i; events that are then skipped because the clause no longer
    fires still used up a place. Dropped events are counted in n_events_dropped_cap.
N3  (5.1) "fires" is re-evaluated once per event. The target step (5.2) and the negative steps
    (5.3) of the same event all run, even when an earlier step of this event (for example a
    Type II inclusion) stopped the clause from firing on this example.
N4  (5.6) SEED_CAP counts SERVED requests per class per batch. A request whose seed is
    abandoned (zero accepted literals) was served and uses up one of the SEED_CAP places.
    A request is served only while the free list is non-empty.
N5  (5.3) In neg_mode = uniform_binomial the veto draw uses `n = c` (the class id), like the
    error gate of that mode, because "the n-th selected negative" is not defined there.
N6  (5.4) `owed / s` with a non-integer `s`: exact rational floor, `owed * den // num` with
    `num / den` the exact value of the float `s`. For integer `s` this is `owed // s`.
N7  (2, 5.4) The state table calls `G` Q32, section 5.4 and 5.5 say Q16. Q16 is implemented.
N8  (6) Tabular input outside dense-equivalent mode is underspecified. Chosen: the index looks
    up literal (0, f, 0) for every f with x[f] == 1; `sym_freq` does not exist, so every
    frequency counts as equal and the anchor is the lowest included positive literal id;
    candidate sampling keeps the three draws t, t+1, t+2: the segment draw has the single
    outcome 0, the kind draw is unused, and the feature is `f = pick(F, ..., t + 2)` (the
    analogue of a bag position); x[f] == 0 means "no candidate" (no retry). Seeds in tabular
    mode get at most one literal, because all literals share segment 0 and 5.6 rejects
    duplicates of an already chosen segment.
N9  (6) Dense-equivalent mode: a literal slot whose state reaches 0 is freed exactly as in
    5.2.1, and with no insertion it never comes back (a standard TM keeps the automaton at its
    lowest state instead). `decay_mark` and `own_fire_mass` are not used by decay_mode =
    eager_exact and stay 0; `G`, `stat_fire`, `stat_tp`, `last_tp_step` are maintained as in
    5.1 and 5.5. There is no sweep in dense-equivalent mode (no recycling). train calls must
    carry exactly one example (B = 1).
N10 (5.4) decay_mode = eager_exact in the sparse engine: for every clause alive at batch start
    and every example i in ascending order, if the clause is NOT in F(i) and it owns y_i
    (live), the target gate `bern(Py; S_GATE_TGT, a, j, 0)` is drawn and, if it passes, every
    used literal is decremented w.p. 1/s (`S_TYPE1_DEC, a, j, k`). Events of F(i) are handled
    as in 5.1. `settle` is a no-op.
N11 (1.4) A literal whose segment is not enabled in this mode (right side in causal mode, bag
    buckets >= N_BAG_K, segments other than 0 in tabular mode) has a FALSE positive form.
    Polarity symbols (reserved) never match.
N12 (api) DIAG_COUNTERS are not defined by the spec. Definitions used here: n_candidates =
    posting-list entries looked up (dense mode: C per example); n_fired = sum of |F(i)|;
    n_events_applied = events that passed the live re-check (dense mode: firing clauses);
    n_type1 = Type Ia applications plus, in eager_exact, non-firing Type I applications;
    n_type2 = Type II applications on firing clauses; n_inserted / n_neg_inserted = literals
    actually placed; n_vetoes = veto votes actually created; n_seed_requests = examples whose
    request draw passed; n_seeded = clauses created; n_recycled = clauses recycled by a sweep.
"""

from __future__ import annotations

import heapq
from bisect import bisect_right, insort
from fractions import Fraction

import numpy as np

from ..api import DIAG_COUNTERS, STATE_FIELDS, CorpusArrays, EngineBase, TabularData, state_shapes
from ..config import (
    NEG_K_MAX,
    ANCHOR_NONE,
    CLS_EMPTY,
    INCLUDE_THRESHOLD,
    LIT_EMPTY,
    N_EXACT_SEG,
    STATE_DENSE_INIT,
    STATE_INSERT,
    STATE_SEED,
    W_MAX,
    EngineConfig,
    Stream,
    bag_of_seg,
    bag_range,
    kind_of,
    lit_pack,
    lit_text,
    lit_unpack,
    seg_offset,
    sym_edge,
    sym_mask,
)
from ..rng import bern, perm, pick, rng32

Q32_MAX = (1 << 32) - 1
U32_MAX = (1 << 32) - 1
NEG_CLASS_DRAW_BASE = 65536  # spec 5.3: selection draw of uniform_binomial uses c = 65536 + class


class _Example:
    """One example (spec 1.3) plus what the forward pass (spec 4) computed for it."""

    __slots__ = (
        "e", "p", "y", "a",            # epoch, target position (or row), target class, uid
        "doc_lo", "doc_hi",            # document of p: doc_lo <= q < doc_hi
        "x",                           # tabular mode: the feature row, list of 0/1
        "ctx",                         # ctx[seg] = (sym kind 0, 1, 2, 3) for enabled exact segments
        "bitmap", "r_a_q32",           # frozen hidden bitmap; diffusion rate of this example
        "hidden_cache", "truth",       # memo tables, no semantics
        "n_candidates", "fired",       # spec 4 steps 2 and 3
        "scores", "s_y",               # spec 4 step 4: {class: S_i[class]}, raw S_i[y]
        "py_q32", "cover", "negatives",  # spec 4 step 5
    )


class OracleEngine(EngineBase):
    """Reference implementation of every mode of docs/11-engine-spec.md."""

    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays | None = None,
                 tabular: TabularData | None = None) -> None:
        super().__init__(cfg, corpus, tabular)
        self.n_clauses = cfg.n_clauses
        self.l_slots = cfg.l_slots
        self.v_slots = cfg.v_slots
        self.n_classes = cfg.vocab
        self.is_tabular = cfg.mode == "tabular"
        self.is_dense = cfg.dense_equivalent
        self.is_lazy = cfg.decay_mode == "lazy_owned" and not cfg.dense_equivalent
        self.is_focused = cfg.neg_mode == "focused" and not cfg.dense_equivalent
        self.is_binomial = cfg.neg_mode == "binomial" and not cfg.dense_equivalent
        # N6: exact rational value of s for the integer division of settle (5.4)
        self.s_num, self.s_den = Fraction(cfg.s).as_integer_ratio()
        self.min_free = cfg.n_clauses // 128  # spec 5.7

        self.sym_mask = sym_mask(cfg.vocab)
        self.sym_edge = sym_edge(cfg.vocab)
        self.enabled_segs = frozenset(self.derived.seg_ids)
        self.exact_segs = self.derived.exact_seg_ids
        if corpus is not None:
            # small static tables as Python lists; tok / doc_start / targets stay numpy (storage)
            self.kind_map = [row.tolist() for row in corpus.kind_map]
            self.stop = corpus.stop.tolist()
            self.sym_freq = corpus.sym_freq.tolist()
            self.n_tok = int(corpus.tok.shape[0])
        self._mask_cache: dict[int, dict[int, bool]] = {}  # masked mode: epoch -> {q: hidden}
        self.diag: dict[str, int] = dict.fromkeys(DIAG_COUNTERS, 0)

        self._init_empty_state()
        if self.is_dense:
            self._init_dense()
        self._rebuild_derived()

    # ------------------------------------------------------------------ randomness (spec 3)
    def _rng32(self, stream: int, a: int, b: int, c: int) -> int:
        return rng32(self.cfg.seed, int(stream), a, b, c)

    def _bern(self, p_q32: int, stream: int, a: int, b: int, c: int) -> bool:
        return bern(p_q32, self.cfg.seed, int(stream), a, b, c)

    def _pick(self, n: int, stream: int, a: int, b: int, c: int) -> int:
        return pick(n, self.cfg.seed, int(stream), a, b, c)

    def _count(self, name: str, n: int = 1) -> None:
        self.diag[name] += n

    # ------------------------------------------------------------------ state (spec 2)
    def _init_empty_state(self) -> None:
        """The INITIAL state: every clause free, every slot cleared, every other field 0."""
        n, nl, nv = self.n_clauses, self.l_slots, self.v_slots
        self.lit_id = [[LIT_EMPTY] * nl for _ in range(n)]
        self.lit_state = [[0] * nl for _ in range(n)]
        self.vote_cls = [[CLS_EMPTY] * nv for _ in range(n)]
        self.vote_w = [[0] * nv for _ in range(n)]
        self.decay_mark = [0] * n
        self.own_fire_mass = [0] * n
        self.stat_fire = [0] * n
        self.stat_tp = [0] * n
        self.birth_step = [0] * n
        self.last_tp_step = [0] * n
        self.alive = [0] * n
        self.G = [0] * self.n_classes
        self.step = 0

    def _init_dense(self) -> None:
        """Spec 6: all 2F literals pre-tracked at 127, every clause alive, one vote per class."""
        for j in range(self.n_clauses):
            for f in range(self.cfg.n_features):
                self.lit_id[j][2 * f] = lit_pack(0, f, 0)
                self.lit_id[j][2 * f + 1] = lit_pack(0, f, 1)
                self.lit_state[j][2 * f] = STATE_DENSE_INIT
                self.lit_state[j][2 * f + 1] = STATE_DENSE_INIT
            for c in range(self.n_classes):
                self.vote_cls[j][c] = c
                self.vote_w[j][c] = 1 if self._rng32(Stream.SEED, j, c, 0) & 1 else -1
            self.alive[j] = 1

    def state(self) -> dict[str, np.ndarray]:
        values = {
            "lit_id": self.lit_id, "lit_state": self.lit_state, "vote_cls": self.vote_cls,
            "vote_w": self.vote_w, "decay_mark": self.decay_mark,
            "own_fire_mass": self.own_fire_mass, "stat_fire": self.stat_fire,
            "stat_tp": self.stat_tp, "birth_step": self.birth_step,
            "last_tp_step": self.last_tp_step, "alive": self.alive, "G": self.G,
            "step": [self.step],
        }
        shapes = state_shapes(self.cfg)
        return {name: np.array(values[name], dtype=dtype).reshape(shapes[name])
                for name, dtype, _ in STATE_FIELDS}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        shapes = state_shapes(self.cfg)
        for name, dtype, _ in STATE_FIELDS:
            arr = state[name]
            if arr.dtype != np.dtype(dtype) or arr.shape != shapes[name]:
                raise ValueError(f"state[{name!r}]: expected {np.dtype(dtype)} {shapes[name]}")
        for name, _, kind in STATE_FIELDS:
            if kind != "1":
                setattr(self, name, state[name].tolist())
        self.step = int(state["step"][0])
        self._rebuild_derived()

    def _rebuild_derived(self) -> None:
        """Derived structures of spec 2: inc_lit, n_inc, anchor, inverted index, free list."""
        n = self.n_clauses
        self.inc_lit: list[list[int]] = [[] for _ in range(n)]
        self.n_inc = [0] * n
        self.n_inc_neg = [0] * n  # included negated literals (neg_budget, docs/11 section 8a N13)
        self.anchor = [ANCHOR_NONE] * n
        self.index: dict[int, list[int]] = {}
        for j in range(n):
            self._refresh(j)
        self.free = [j for j in range(n) if not self.alive[j]]  # ascending, so a valid heap

    def _refresh(self, j: int) -> None:
        """refresh(j) of spec 2: recompute inc_lit, n_inc and anchor of clause j, fix the index."""
        groups: tuple[list[int], ...] = ([], [], [], [])  # exact pos, exact neg, bag pos, bag neg
        if self.alive[j]:
            ids, states = self.lit_id[j], self.lit_state[j]
            for k in range(self.l_slots):
                if states[k] >= INCLUDE_THRESHOLD and ids[k] != LIT_EMPTY:
                    seg, _, neg = lit_unpack(ids[k])
                    groups[(2 if seg >= N_EXACT_SEG else 0) + neg].append(ids[k])
        self.inc_lit[j] = [lit for group in groups for lit in sorted(group)]
        self.n_inc[j] = len(self.inc_lit[j])
        self.n_inc_neg[j] = len(groups[1]) + len(groups[3])
        new_anchor = self._anchor_of(groups[0])
        old_anchor = self.anchor[j]
        if new_anchor != old_anchor:
            if old_anchor != ANCHOR_NONE:
                self.index[old_anchor].remove(j)
                if not self.index[old_anchor]:
                    del self.index[old_anchor]
            if new_anchor != ANCHOR_NONE:
                insort(self.index.setdefault(new_anchor, []), j)
            self.anchor[j] = new_anchor

    def _freq(self, sym: int) -> int:
        """Training frequency of a symbol for the anchor choice (spec 2)."""
        if self.is_tabular:
            return 0  # N8: no sym_freq in tabular mode, all frequencies equal
        if sym == self.sym_mask or sym == self.sym_edge:
            return U32_MAX
        return self.sym_freq[sym]

    def _anchor_of(self, positive_exact_lits: list[int]) -> int:
        """Anchor (spec 2): smallest sym_freq, ties toward the lower literal id."""
        best_key, best = None, ANCHOR_NONE
        for lit in positive_exact_lits:
            key = (self._freq(lit_unpack(lit)[1]), lit)
            if best_key is None or key < best_key:
                best_key, best = key, lit
        return best

    # vote helpers -------------------------------------------------------------------------
    def _vote_slot(self, j: int, c: int) -> int | None:
        """Slot index of class c in clause j (class ids are unique within a clause), or None."""
        try:
            return self.vote_cls[j].index(c)
        except ValueError:
            return None

    def _vote(self, j: int, c: int) -> int:
        """Vote of clause j for class c, 0 if there is no slot."""
        k = self._vote_slot(j, c)
        return 0 if k is None else self.vote_w[j][k]

    def _owns(self, j: int, c: int) -> bool:
        """owns(j, c) of spec 2."""
        return self._vote(j, c) > 0

    def _mass(self, j: int) -> int:
        """M(j) of spec 5.4: sum of G[c] over the classes that clause j owns."""
        cls, w = self.vote_cls[j], self.vote_w[j]
        return sum(self.G[cls[k]] for k in range(self.v_slots) if cls[k] != CLS_EMPTY and w[k] > 0)

    # ------------------------------------------------------------------ examples (spec 1.3, 6)
    def _make_example(self, epoch: int, ident: int, bitmap: np.ndarray | None) -> _Example:
        ex = _Example()
        ex.e, ex.p, ex.a = epoch, ident, (epoch << 40) | ident
        ex.truth, ex.hidden_cache = {}, {}
        ex.bitmap, ex.r_a_q32, ex.x, ex.ctx = None, 0, None, None
        ex.doc_lo = ex.doc_hi = 0
        if self.is_tabular:
            if not 0 <= ident < self.n_examples:
                raise ValueError("row index out of range")
            ex.x = self.tabular.x[ident].tolist()
            ex.y = int(self.tabular.y[ident])
            return ex
        if not 0 <= ident < self.n_tok:
            raise ValueError("position out of range")
        ex.y = int(self.corpus.tok[ident])
        d = int(np.searchsorted(self.corpus.doc_start, ident, side="right")) - 1
        ex.doc_lo, ex.doc_hi = int(self.corpus.doc_start[d]), int(self.corpus.doc_start[d + 1])
        if self.cfg.mode in ("masked", "diffusion"):
            ex.bitmap = bitmap
            if self.cfg.mode == "diffusion" and bitmap is None:
                ex.r_a_q32 = self._rng32(Stream.RATE, ex.a, 0, 0)
        ex.ctx = [None] * N_EXACT_SEG
        for seg in self.exact_segs:
            q = ident + seg_offset(seg)
            ex.ctx[seg] = tuple(self._ctx(ex, q, k) for k in range(4))
        return ex

    def _hidden(self, ex: _Example, q: int) -> bool:
        """hidden(a, q) of spec 1.3. The target position itself is handled in _ctx."""
        if ex.bitmap is not None:
            return int(ex.bitmap[q]) != 0
        mode = self.cfg.mode
        if mode == "masked":
            cache = self._mask_cache.get(ex.e)
            if cache is None:
                if len(self._mask_cache) >= 4:
                    self._mask_cache.clear()
                cache = self._mask_cache.setdefault(ex.e, {})
            h = cache.get(q)
            if h is None:
                h = cache[q] = self._rng32(Stream.MASK, ex.e, q, 0) < self.derived.mask_rate_q32
            return h
        if mode == "diffusion":
            h = ex.hidden_cache.get(q)
            if h is None:
                h = ex.hidden_cache[q] = self._rng32(Stream.MASK, ex.a, q, 0) < ex.r_a_q32
            return h
        return False  # causal: nothing is hidden

    def _ctx(self, ex: _Example, q: int, k: int) -> int:
        """ctx(a, q, k) of spec 1.3."""
        if q < ex.doc_lo or q >= ex.doc_hi:
            return self.sym_edge
        if q == ex.p or self._hidden(ex, q):
            return self.sym_mask
        return self.kind_map[k][int(self.corpus.tok[q])]

    # ------------------------------------------------------------------ literal truth (spec 1.4)
    def _lit_true(self, ex: _Example, lit: int) -> bool:
        v = ex.truth.get(lit)
        if v is None:
            seg, sym, neg = lit_unpack(lit)
            v = ex.truth[lit] = self._positive_true(ex, seg, sym) != bool(neg)
        return v

    def _positive_true(self, ex: _Example, seg: int, sym: int) -> bool:
        if seg not in self.enabled_segs:
            return False  # N11
        if self.is_tabular:
            return sym < self.cfg.n_features and ex.x[sym] == 1
        if seg < N_EXACT_SEG:
            return self._exact_true(ex, seg, sym)
        return self._bag_true(ex, seg, sym)

    def _exact_true(self, ex: _Example, seg: int, sym: int) -> bool:
        if sym == self.sym_mask or sym == self.sym_edge:
            return ex.ctx[seg][0] == sym
        kind = kind_of(self.cfg.vocab, sym)
        return kind >= 0 and ex.ctx[seg][kind] == sym

    def _bag_true(self, ex: _Example, seg: int, sym: int) -> bool:
        """The literal definition of spec 1.4: scan the distance range of the bucket."""
        kind = kind_of(self.cfg.vocab, sym)
        if kind < 0:
            return False
        bucket, side = bag_of_seg(seg)
        lo, hi = bag_range(bucket)
        first, last = (ex.p - hi, ex.p - lo) if side == 0 else (ex.p + lo, ex.p + hi)
        first, last = max(first, ex.doc_lo), min(last, ex.doc_hi - 1)
        if first > last:
            return False
        kmap = self.kind_map[kind]
        toks = self.corpus.tok[first:last + 1].tolist()
        for q, t in zip(range(first, last + 1), toks):
            if kmap[t] == sym and self.stop[t] == 0 and not self._hidden(ex, q):
                return True
        return False

    # ------------------------------------------------------------------ forward pass (spec 4)
    def _candidates(self, ex: _Example) -> list[int]:
        """Spec 4 step 2. Every literal is looked up at most once."""
        out: list[int] = []
        if self.is_tabular:  # N8
            for f, bit in enumerate(ex.x):
                if bit == 1:
                    out.extend(self.index.get(lit_pack(0, f, 0), ()))
            return out
        for seg in self.exact_segs:
            row = ex.ctx[seg]
            syms = (row[0],) if row[0] in (self.sym_mask, self.sym_edge) else row
            for sym in syms:
                out.extend(self.index.get(lit_pack(seg, sym, 0), ()))
        return out

    def _fires(self, ex: _Example, j: int) -> bool:
        """Spec 2 / 4 step 3, against the refreshed (batch start) inc_lit of clause j."""
        if not self.alive[j] or self.anchor[j] == ANCHOR_NONE:
            return False
        return all(self._lit_true(ex, lit) for lit in self.inc_lit[j])

    def _fires_live(self, ex: _Example, j: int) -> bool:
        """The same predicate computed from the slot arrays only (live state, no index)."""
        if not self.alive[j]:
            return False
        ids, states = self.lit_id[j], self.lit_state[j]
        has_anchor = False
        for k in range(self.l_slots):
            if states[k] >= INCLUDE_THRESHOLD and ids[k] != LIT_EMPTY:
                if not self._lit_true(ex, ids[k]):
                    return False
                if ids[k] & 1 == 0 and (ids[k] >> 17) < N_EXACT_SEG:
                    has_anchor = True
        return has_anchor

    def _fires_dense(self, ex: _Example, j: int, training: bool) -> bool:
        """Spec 6: no anchor requirement; an empty clause fires in training, not at prediction."""
        ids, states = self.lit_id[j], self.lit_state[j]
        n_included = 0
        for k in range(self.l_slots):
            if states[k] >= INCLUDE_THRESHOLD and ids[k] != LIT_EMPTY:
                if not self._lit_true(ex, ids[k]):
                    return False
                n_included += 1
        return training or n_included > 0

    def _fired(self, ex: _Example, training: bool) -> list[int]:
        """Spec 4 steps 2 and 3: F(i), ascending clause ids."""
        if self.is_dense:
            ex.n_candidates = self.n_clauses
            return [j for j in range(self.n_clauses) if self._fires_dense(ex, j, training)]
        cands = self._candidates(ex)
        ex.n_candidates = len(cands)
        return sorted(j for j in cands if self._fires(ex, j))

    def _scores(self, fired: list[int]) -> dict[int, int]:
        """Spec 4 step 4: S_i[c] = sum of w over fired clauses and used vote slots of class c."""
        scores: dict[int, int] = {}
        for j in fired:
            cls, w = self.vote_cls[j], self.vote_w[j]
            for k in range(self.v_slots):
                if cls[k] != CLS_EMPTY:
                    scores[cls[k]] = scores.get(cls[k], 0) + w[k]
        return scores

    def _clamp_t(self, s: int) -> int:
        return max(-self.cfg.T, min(self.cfg.T, s))

    def _py_q32(self, s_y: int) -> int:
        t = self.cfg.T
        return min(Q32_MAX, ((t - self._clamp_t(s_y)) << 32) // (2 * t))

    def _pc_q32(self, s_c: int) -> int:
        t = self.cfg.T
        return min(Q32_MAX, ((t + self._clamp_t(s_c)) << 32) // (2 * t))

    def _forward(self, ex: _Example) -> None:
        """Spec 4 steps 2 to 5 for one example, against the state at batch start."""
        self._forward_partial(ex)
        self._forward_finish(ex)

    def _forward_partial(self, ex: _Example) -> None:
        """Spec 4 steps 2 to 4 over the clauses THIS engine holds: fired set, (partial) scores and
        (partial) cover. In a sharded run (spec section 9) the shards' scores and covers are summed
        by exchange 1 before `_forward_finish`."""
        ex.fired = self._fired(ex, training=True)
        ex.scores = self._scores(ex.fired)
        ex.cover = sum(1 for j in ex.fired if self._owns(j, ex.y))

    def _forward_finish(self, ex: _Example) -> None:
        """Spec 4 step 5 from the (summed) scores: S_y, the target gate and the ranked negatives.
        A pure function of `ex.scores`, so identical on every shard after exchange 1."""
        ex.s_y = ex.scores.get(ex.y, 0)
        ex.py_q32 = self._py_q32(ex.s_y)
        # negatives: up to K_NEG classes c != y with the largest S > 0, ties toward lower class id
        ranked = sorted((-s, c) for c, s in ex.scores.items() if c != ex.y and s > 0)
        ex.negatives = [(c, self._pc_q32(-neg_s), -neg_s) for neg_s, c in ranked[:self.cfg.k_neg]]

    # ------------------------------------------------------------------ API: prediction
    def _ids(self, examples) -> list[int]:
        return [int(v) for v in np.asarray(examples).reshape(-1)]

    def _bitmap(self, hidden: np.ndarray | None) -> np.ndarray | None:
        if hidden is None or self.cfg.mode not in ("masked", "diffusion"):
            return None
        if hidden.shape != (self.n_tok,):
            raise ValueError("hidden must be a bitmap over all token positions")
        return hidden

    def predict_scores(self, epoch: int, examples: np.ndarray,
                       hidden: np.ndarray | None = None) -> np.ndarray:
        ids = self._ids(examples)
        out = np.zeros((len(ids), self.n_classes), dtype=np.int32)
        bitmap = self._bitmap(hidden)
        for row, ident in enumerate(ids):
            ex = self._make_example(epoch, ident, bitmap)
            for c, s in self._scores(self._fired(ex, training=False)).items():
                out[row, c] = s
        return out

    def fired_clauses(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        ex = self._make_example(epoch, int(example), self._bitmap(hidden))
        return np.array(self._fired(ex, training=False), dtype=np.uint32)

    def fired_clauses_bruteforce(self, epoch: int, example: int,
                                 hidden: np.ndarray | None = None) -> np.ndarray:
        """Like fired_clauses, but evaluates every clause from the slot arrays, without the index,
        inc_lit or anchor. Used by the index-consistency test."""
        ex = self._make_example(epoch, int(example), self._bitmap(hidden))
        if self.is_dense:
            fired = [j for j in range(self.n_clauses) if self._fires_dense(ex, j, training=False)]
        else:
            fired = [j for j in range(self.n_clauses) if self._fires_live(ex, j)]
        return np.array(fired, dtype=np.uint32)

    # ------------------------------------------------------------------ API: training (spec 5)
    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        n, b = self.n_examples, self.cfg.batch
        lo = batch_index * b
        if not 0 <= lo < n:
            raise ValueError("batch_index out of range")
        order = [perm(self.cfg.seed, int(Stream.PERM), epoch, x, n) for x in range(lo, min(n, lo + b))]
        if self.is_tabular:
            return self._train(epoch, order)
        return self._train(epoch, [int(self.corpus.targets[idx]) for idx in order])

    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        return self._train(epoch, self._ids(examples))

    def _train(self, epoch: int, ids: list[int]) -> dict[str, int]:
        """One batch in the normative order of spec 5."""
        self.diag = dict.fromkeys(DIAG_COUNTERS, 0)
        examples = [self._make_example(epoch, ident, None) for ident in ids]
        self._count("n_examples", len(examples))
        if self.is_dense:
            self._train_dense(examples)
        else:
            self._train_sparse(examples)
        self.step += 1
        return dict(self.diag)

    def _train_sparse(self, examples: list[_Example]) -> None:
        """One batch as the composition of the phases of spec 5; the sharded oracle
        (oracle/shard.py) calls the same phases with the exchanges of spec section 9 in between."""
        events = self._phase_forward(examples)  # spec 4
        changed = self._phase_feedback(events, examples)  # spec 5.1 to 5.4
        self._gate_mass(examples)  # spec 5.5
        changed |= self._seed(examples)  # spec 5.6
        self._phase_refresh(changed)
        if self.sweep_due():  # N1
            self._sweep()

    def sweep_due(self) -> bool:
        return (self.step + 1) % self.cfg.sweep_period == 0

    def _phase_forward(self, examples: list[_Example]) -> dict[int, list[int]]:
        """Spec 4: forward pass of the whole batch against the state at batch start. Returns the
        touched clauses -> their example indices, ascending i."""
        events: dict[int, list[int]] = {}
        for i, ex in enumerate(examples):
            self._forward(ex)
            self._count("n_candidates", ex.n_candidates)
            self._count("n_fired", len(ex.fired))
            for j in ex.fired:
                events.setdefault(j, []).append(i)
        return events

    def _phase_feedback(self, events: dict[int, list[int]], examples: list[_Example]) -> set[int]:
        """Spec 5.1 to 5.4: feedback, one clause at a time (clauses do not interact). Returns the
        changed clause ids."""
        if self.cfg.decay_mode == "lazy_owned":
            changed = set(events)
            for j in sorted(events):
                self._clause_events(j, events[j], examples)
        else:
            changed = {j for j in range(self.n_clauses) if self.alive[j]}
            for j in sorted(changed):
                self._clause_events_eager(j, events.get(j, []), examples)
        return changed

    def _phase_refresh(self, changed: set[int]) -> None:
        for j in sorted(changed):  # refresh of changed clauses and index rebuild
            self._refresh(j)

    # spec 5.1 -----------------------------------------------------------------------------
    def _clause_events(self, j: int, event_list: list[int], examples: list[_Example]) -> None:
        """Touched clause j: settle first (G as of batch start), then its events in ascending i."""
        self._settle(j)
        cap = self.cfg.e_cap
        for i in event_list[:cap]:  # N2
            self._event(j, examples[i])
        self._count("n_events_dropped_cap", max(0, len(event_list) - cap))

    def _clause_events_eager(self, j: int, event_list: list[int], examples: list[_Example]) -> None:
        """N10: decay_mode = eager_exact. Non-firing owned examples get standard Type Ib feedback."""
        place = {i: n for n, i in enumerate(event_list)}
        for i, ex in enumerate(examples):
            n = place.get(i)
            if n is None:
                if self._owns(j, ex.y) and self._bern(ex.py_q32, Stream.GATE_TGT, ex.a, j, 0):
                    self._count("n_tgt_gate_pass")
                    self._type_ib(j, ex)
            elif n < self.cfg.e_cap:
                self._event(j, ex)
            else:
                self._count("n_events_dropped_cap")

    def _event(self, j: int, ex: _Example) -> None:
        """Spec 5.1: event (i, j) against the LIVE state of clause j."""
        if not self._fires_live(ex, j):
            self._count("n_events_skipped_not_firing")
            return
        self._count("n_events_applied")
        self._fire_stats(j, ex)
        self._target_feedback(j, ex)  # N3: both steps run once the event is accepted
        self._negative_feedback(j, ex)

    def _fire_stats(self, j: int, ex: _Example) -> None:
        self.stat_fire[j] = min(U32_MAX, self.stat_fire[j] + 1)
        if self._owns(j, ex.y):
            self.stat_tp[j] = min(U32_MAX, self.stat_tp[j] + 1)
            self.last_tp_step[j] = self.step & U32_MAX
            if self.is_lazy:
                self.own_fire_mass[j] += ex.py_q32 >> 16  # Q16, spec 5.5

    # spec 5.2 -----------------------------------------------------------------------------
    def _target_feedback(self, j: int, ex: _Example) -> None:
        if not self._bern(ex.py_q32, Stream.GATE_TGT, ex.a, j, 0):
            return
        self._count("n_tgt_gate_pass")
        w = self._vote(j, ex.y)
        if w > 0:
            self._type_ia(j, ex)
            self._vote_add(j, ex.y, +1)
        elif w < 0:
            self._type_ii(j, ex)
            self._vote_add(j, ex.y, +1)
        elif self._vote_add(j, ex.y, +1):  # no slot: vote first, Type Ia only with a slot
            self._type_ia(j, ex)

    def _vote_add(self, j: int, c: int, delta: int) -> bool:
        """vote_add(j, c, delta) of spec 5.2. Returns True iff clause j has (or had, if the vote
        just reached 0) a slot for c, that is: False only when no slot could be obtained."""
        cls, w = self.vote_cls[j], self.vote_w[j]
        k = self._vote_slot(j, c)
        if k is not None:
            new_w = max(-W_MAX, min(W_MAX, w[k] + delta))
            change = (w[k] > 0) != (new_w > 0)
            if change:
                self._settle(j)  # ownership-change protocol: settle BEFORE the change
            w[k] = new_w
            if new_w == 0:
                cls[k] = CLS_EMPTY
            if change:
                self._mark(j)
            return True
        k = self._lowest_free_vote_slot(j)
        if k is None:
            k = min(range(self.v_slots), key=lambda q: (abs(w[q]), q))
            if abs(w[k]) > 1:
                return False
        change = delta > 0 or w[k] > 0  # a positive vote appears, or an owned class is evicted
        if change:
            self._settle(j)
        cls[k], w[k] = c, delta
        if change:
            self._mark(j)
        return True

    def _lowest_free_vote_slot(self, j: int) -> int | None:
        cls = self.vote_cls[j]
        for k in range(self.v_slots):
            if cls[k] == CLS_EMPTY:
                return k
        return None

    def _mark(self, j: int) -> None:
        """Last step of the ownership-change protocol (5.4): decay_mark = M(j), new ownership."""
        if self.is_lazy:
            self.decay_mark[j] = self._mass(j)

    # spec 5.2.1 ---------------------------------------------------------------------------
    def _type_ia(self, j: int, ex: _Example) -> None:
        self._count("n_type1")
        ids = self.lit_id[j]
        for k in range(self.l_slots):
            if ids[k] == LIT_EMPTY:
                continue
            if self._lit_true(ex, ids[k]):
                self._state_up(j, k)
            elif self._bern(self.derived.inv_s_q32, Stream.TYPE1_DEC, ex.a, j, k):
                self._state_down(j, k, 1)
        if self._bern(self.derived.p_ins_q32, Stream.INSERT, ex.a, j, 0):
            lit = self._sample_candidate(ex, Stream.INSERT_PICK, j, 0, exact_only=False)
            if lit is not None and self._insert_literal(j, lit):
                self._count("n_inserted")

    def _type_ib(self, j: int, ex: _Example) -> None:
        """Standard Type I feedback on a NON-firing clause (eager_exact and spec 6 only)."""
        self._count("n_type1")
        ids = self.lit_id[j]
        for k in range(self.l_slots):
            if ids[k] != LIT_EMPTY and self._bern(self.derived.inv_s_q32, Stream.TYPE1_DEC, ex.a, j, k):
                self._state_down(j, k, 1)

    def _state_up(self, j: int, k: int) -> None:
        """state = min(255, state + 1); the 127 -> 128 crossing is suppressed at BUDGET, and for a
        negated literal also at NEG_BUDGET (N13)."""
        s = self.lit_state[j][k]
        if s == 255:
            return
        if s == INCLUDE_THRESHOLD - 1:
            neg = self.lit_id[j][k] & 1
            if not self.is_dense and self.n_inc[j] >= self.cfg.budget:
                return
            if not self.is_dense and neg and 0 <= self.cfg.neg_budget <= self.n_inc_neg[j]:
                return
            self.n_inc[j] += 1  # n_inc is updated immediately on a crossing
            self.n_inc_neg[j] += neg
        self.lit_state[j][k] = s + 1

    def _state_down(self, j: int, k: int, dec: int) -> None:
        """state = max(0, state - dec); a slot reaching 0 is freed."""
        s = self.lit_state[j][k]
        new_s = max(0, s - dec)
        if s >= INCLUDE_THRESHOLD > new_s:
            self.n_inc[j] -= 1
            self.n_inc_neg[j] -= self.lit_id[j][k] & 1
        self.lit_state[j][k] = new_s
        if new_s == 0:
            self.lit_id[j][k] = LIT_EMPTY

    def _insert_literal(self, j: int, lit: int) -> bool:
        """Slot rule of 5.2.1: lowest free slot, else evict the lowest EXCLUDED state if < 123."""
        ids, states = self.lit_id[j], self.lit_state[j]
        if lit in ids:
            return False  # already tracked
        slot = None
        for k in range(self.l_slots):
            if ids[k] == LIT_EMPTY:
                slot = k
                break
        if slot is None:
            excluded = [k for k in range(self.l_slots) if states[k] < INCLUDE_THRESHOLD]
            if not excluded:
                return False
            slot = min(excluded, key=lambda k: (states[k], k))
            if states[slot] >= STATE_INSERT:
                return False
        ids[slot], states[slot] = lit, STATE_INSERT
        return True

    # spec 5.2.2 ---------------------------------------------------------------------------
    def _type_ii(self, j: int, ex: _Example) -> None:
        self._count("n_type2")
        ids, states = self.lit_id[j], self.lit_state[j]
        for k in range(self.l_slots):
            if ids[k] != LIT_EMPTY and states[k] < INCLUDE_THRESHOLD and not self._lit_true(ex, ids[k]):
                self._state_up(j, k)
        if self._bern(self.derived.p_neg_q32, Stream.NEG_INSERT, ex.a, j, 0):
            lit = self._sample_candidate(ex, Stream.NEG_PICK, j, 0, exact_only=True)
            if lit is not None and self._insert_literal(j, lit | 1):  # the NEGATION of the candidate
                self._count("n_neg_inserted")

    # spec 5.2.3 ---------------------------------------------------------------------------
    def _sample_candidate(self, ex: _Example, stream: int, j: int, t: int, exact_only: bool) -> int | None:
        """N15 (spec 8d): `cand_tries` single draws (try i uses base t + 9 * i), keeping the candidate
        with the smallest symbol frequency (ties toward the earlier try); None if every try is empty."""
        best, best_key = None, None
        for i in range(self.cfg.cand_tries):
            lit = self._sample_candidate_once(ex, stream, j, t + 9 * i, exact_only)
            if lit is None:
                continue
            key = self._freq(lit_unpack(lit)[1])
            if best_key is None or key < best_key:
                best, best_key = lit, key
        return best

    def _sample_candidate_once(self, ex: _Example, stream: int, j: int, t: int, exact_only: bool) -> int | None:
        """One candidate literal, TRUE and POSITIVE on the example, or None ("no candidate").
        Draws t, t + 1, t + 2 of the stream select segment, kind and bag position."""
        d = self.derived
        seg_ids, cum, total = ((d.exact_seg_ids, d.exact_cum, d.exact_total) if exact_only
                               else (d.seg_ids, d.seg_cum, d.seg_total))
        seg = seg_ids[bisect_right(cum, self._pick(total, stream, ex.a, j, t))]  # first cum > x
        if self.is_tabular:  # N8
            f = self._pick(self.cfg.n_features, stream, ex.a, j, t + 2)
            return lit_pack(seg, f, 0) if ex.x[f] == 1 else None
        u = self._rng32(stream, ex.a, j, t + 1)
        if seg < N_EXACT_SEG:
            kind = 0 if u < d.kind_t70 else 1 if u < d.kind_t80 else 2 if u < d.kind_t90 else 3
            sym = ex.ctx[seg][kind]
            return None if sym == self.sym_mask else lit_pack(seg, sym, 0)
        kind = 0 if u < d.kind_t70 else 2
        bucket, side = bag_of_seg(seg)
        lo, hi = bag_range(bucket)
        dist = lo + self._pick(hi - lo + 1, stream, ex.a, j, t + 2)
        q = ex.p - dist if side == 0 else ex.p + dist
        if q < ex.doc_lo or q >= ex.doc_hi or self._hidden(ex, q):
            return None
        tok = int(self.corpus.tok[q])
        if self.stop[tok] != 0:
            return None
        return lit_pack(seg, self.kind_map[kind][tok], 0)

    # spec 5.3 -----------------------------------------------------------------------------
    def draw_negative_set(self, a: int, j: int, y: int) -> list[int]:
        """Spec 5.3, neg_mode = binomial (N14): k ~ Binomial(V-1, Q/(V-1)) by inversion of the Q32 CDF
        table with u = rng32(S_NEG_DRAW, a, j, 0), then k DISTINCT classes != y, class number i
        drawn as pick(V-1; S_NEG_DRAW, a, j, 1 + t) with t counting every attempt (a duplicate is
        rejected and t advances; at most 8 * NEG_K_MAX attempts); pick c >= y maps to c + 1. The set
        of selected classes has exactly the distribution of independent Bernoulli(Q/(V-1)) per
        class truncated at NEG_K_MAX, which is the dense Q gate of uniform_binomial mode."""
        u = self._rng32(Stream.NEG_DRAW, a, j, 0)
        cdf = self.derived.neg_cdf_q32
        k = 0
        while k < NEG_K_MAX and cdf[k] <= u:
            k += 1
        chosen: list[int] = []
        t = 0
        while len(chosen) < k and t < 8 * NEG_K_MAX:
            c = self._pick(self.n_classes - 1, Stream.NEG_DRAW, a, j, 1 + t)
            t += 1
            if c >= y:
                c += 1
            if c not in chosen:
                chosen.append(c)
        return chosen

    def _negative_feedback(self, j: int, ex: _Example) -> None:
        if self.is_focused:
            for n, (c, pc_q32, s_c) in enumerate(ex.negatives):
                if self._bern(pc_q32, Stream.GATE_NEG, ex.a, j, n):
                    self._negative_update(j, ex, c, s_c, n)
            return
        if self.is_binomial:
            for c in self.draw_negative_set(ex.a, j, ex.y):  # selection order = draw order
                s_c = ex.scores.get(c, 0)
                if self._bern(self._pc_q32(s_c), Stream.GATE_NEG, ex.a, j, c):
                    self._negative_update(j, ex, c, s_c, c)  # N5: veto n = c
            return
        for c in range(self.n_classes):  # neg_mode = uniform_binomial: dense gate per (clause, class)
            if c == ex.y:
                continue
            if not self._bern(self.derived.q_neg_q32, Stream.GATE_NEG, ex.a, j, NEG_CLASS_DRAW_BASE + c):
                continue
            s_c = ex.scores.get(c, 0)
            if self._bern(self._pc_q32(s_c), Stream.GATE_NEG, ex.a, j, c):
                self._negative_update(j, ex, c, s_c, c)  # N5

    def _negative_update(self, j: int, ex: _Example, c: int, s_c: int, n: int) -> None:
        self._count("n_neg_gate_pass")
        w = self._vote(j, c)
        if w > 0:
            self._type_ii(j, ex)
            self._vote_add(j, c, -1)
        elif w < 0:
            self._type_ia(j, ex)
            self._vote_add(j, c, -1)
        elif s_c >= ex.s_y - self.derived.margin:
            if self._bern(self.derived.p_veto_q32, Stream.VETO, ex.a, j, n):
                k = self._lowest_free_vote_slot(j)  # ONLY into a free slot, never by eviction
                if k is not None:
                    self.vote_cls[j][k], self.vote_w[j][k] = c, -1
                    self._count("n_vetoes")

    # spec 5.4 -----------------------------------------------------------------------------
    def _settle(self, j: int) -> None:
        if not self.is_lazy:
            return
        m = self._mass(j)
        owed = max(0, m - self.decay_mark[j] - self.own_fire_mass[j])
        n_q16 = owed * self.s_den // self.s_num  # N6: owed / s, integer division
        n_int = n_q16 >> 16
        frac_q32 = (n_q16 & 0xFFFF) << 16
        if n_q16 > 0:  # with n_q16 == 0 every dec below is 0
            ids = self.lit_id[j]
            for k in range(self.l_slots):
                if ids[k] == LIT_EMPTY:
                    continue
                dec = n_int + (1 if self._bern(frac_q32, Stream.DECAY, self.step, j, k) else 0)
                if dec:
                    self._state_down(j, k, dec)
        self.decay_mark[j] = m
        self.own_fire_mass[j] = 0

    # spec 5.5 -----------------------------------------------------------------------------
    def _gate_mass(self, examples: list[_Example]) -> None:
        for ex in examples:
            self.G[ex.y] += ex.py_q32 >> 16

    # spec 5.6 -----------------------------------------------------------------------------
    def _seed(self, examples: list[_Example]) -> set[int]:
        served: dict[int, int] = {}
        created: set[int] = set()
        for ex in examples:  # ascending i
            if not self._seed_request(ex):
                continue
            if not self.free or served.get(ex.y, 0) >= self.cfg.seed_cap:
                continue
            served[ex.y] = served.get(ex.y, 0) + 1  # N4: a served request counts, even if abandoned
            lits = self._seed_literals(ex)
            if not lits:
                continue  # abandoned, the clause stays free
            j = heapq.heappop(self.free)  # lowest free clause id first
            self._seed_serve(ex, j, lits)
            created.add(j)
        return created

    def _seed_request(self, ex: _Example) -> bool:
        """Spec 5.6: does example i request a seed clause? (cover below K_COVER, then a coin with
        probability P_y * RHO_SEED.) Counts the request."""
        if ex.cover >= self.cfg.k_cover:
            return False
        p_req_q32 = (ex.py_q32 * self.derived.rho_seed_q32) >> 32
        if not self._bern(p_req_q32, Stream.SEED, ex.a, 0, 0):
            return False
        self._count("n_seed_requests")
        return True

    def _seed_serve(self, ex: _Example, j: int, lits: list[int]) -> None:
        """Spec 5.6: turn the free clause j into the seed clause of example i with literals `lits`."""
        for k, lit in enumerate(lits):
            self.lit_id[j][k], self.lit_state[j][k] = lit, STATE_SEED
        self.vote_cls[j][0], self.vote_w[j][0] = ex.y, 1
        self.alive[j] = 1
        self.birth_step[j] = self.last_tp_step[j] = self.step & U32_MAX
        self.stat_fire[j] = self.stat_tp[j] = 0
        self.decay_mark[j] = self.G[ex.y] if self.is_lazy else 0  # G after 5.5
        self.own_fire_mass[j] = 0
        self._count("n_seeded")

    def _seed_literals(self, ex: _Example) -> list[int]:
        u = self._rng32(Stream.SEED_PICK, ex.a, 0, 100)
        n_lit = 1 if u < self.derived.nlit_t1 else 2 if u < self.derived.nlit_t2 else 3
        lits: list[int] = []
        for m in range(n_lit):
            lit = self._sample_candidate(ex, Stream.SEED_PICK, 0, 3 * m, exact_only=True)
            if lit is None or any(lit >> 17 == chosen >> 17 for chosen in lits):
                continue  # SYM_MASK or a duplicate segment: the draw is simply lost
            lits.append(lit)
        return lits

    # spec 5.7 -----------------------------------------------------------------------------
    def _age(self, since_u32: int) -> int:
        return ((self.step & U32_MAX) - since_u32) & U32_MAX  # N1

    def _owns_any(self, j: int) -> bool:
        return any(self.vote_cls[j][k] != CLS_EMPTY and self.vote_w[j][k] > 0
                   for k in range(self.v_slots))

    def _sweep(self) -> None:
        """Spec 5.7 for one engine holding every clause: local recycling, the MIN_FREE top-up over
        its own candidates, the stat halving. The sharded oracle runs the same three pieces with
        exchange 3 (the merged candidate lists) between the first two."""
        self._sweep_local()
        if len(self.free) < self.min_free:
            for _, j in self._sweep_candidates():
                if len(self.free) >= self.min_free:
                    break
                self._recycle(j)
        self._sweep_shift()

    def _sweep_local(self) -> None:
        period = self.cfg.sweep_period
        for j in range(self.n_clauses):
            if not self.alive[j]:
                continue
            self._settle(j)
            self._refresh(j)
            if (self.anchor[j] == ANCHOR_NONE or not self._owns_any(j)
                    or self._age(self.last_tp_step[j]) > self.cfg.dead_sweeps * period):
                self._recycle(j)

    def _sweep_candidates(self) -> list[tuple[int, int]]:
        """The top-up candidates of spec 5.7: alive clauses older than 2 * SWEEP_PERIOD, sorted by
        (stat_tp, id) ascending."""
        period = self.cfg.sweep_period
        return sorted((self.stat_tp[j], j) for j in range(self.n_clauses)
                      if self.alive[j] and self._age(self.birth_step[j]) > 2 * period)

    def _sweep_shift(self) -> None:
        for j in range(self.n_clauses):
            self.stat_fire[j] >>= 1
            self.stat_tp[j] >>= 1

    def _recycle(self, j: int) -> None:
        """Restore exactly the cleared initial form of clause j."""
        self.lit_id[j] = [LIT_EMPTY] * self.l_slots
        self.lit_state[j] = [0] * self.l_slots
        self.vote_cls[j] = [CLS_EMPTY] * self.v_slots
        self.vote_w[j] = [0] * self.v_slots
        self.decay_mark[j] = self.own_fire_mass[j] = 0
        self.stat_fire[j] = self.stat_tp[j] = 0
        self.birth_step[j] = self.last_tp_step[j] = 0
        self.alive[j] = 0
        self._refresh(j)
        heapq.heappush(self.free, j)
        self._count("n_recycled")

    # ------------------------------------------------------------------ spec 6: dense-equivalent
    def _train_dense(self, examples: list[_Example]) -> None:
        if len(examples) != 1:
            raise ValueError("dense_equivalent mode is defined for B = 1 only")  # N9
        ex = examples[0]
        self._forward(ex)  # brute force over all clauses, training sense of "fires"
        self._count("n_candidates", ex.n_candidates)
        self._count("n_fired", len(ex.fired))
        fired = set(ex.fired)
        for j in range(self.n_clauses):
            self._dense_clause_feedback(j, ex, j in fired)
        self._gate_mass(examples)

    def _dense_clause_feedback(self, j: int, ex: _Example, fires: bool) -> None:
        """Standard Coalesced TM feedback for one (example, clause) pair (spec 6)."""
        if fires:
            self._count("n_events_applied")
            self._fire_stats(j, ex)
        if self._bern(ex.py_q32, Stream.GATE_TGT, ex.a, j, 0):  # target class
            self._count("n_tgt_gate_pass")
            self._dense_update(j, ex, ex.y, +1, fires)
        for c in range(self.n_classes):  # drawn negative classes
            if c == ex.y:
                continue
            if not self._bern(self.derived.q_neg_q32, Stream.GATE_NEG, ex.a, j, NEG_CLASS_DRAW_BASE + c):
                continue
            if self._bern(self._pc_q32(ex.scores.get(c, 0)), Stream.GATE_NEG, ex.a, j, c):
                self._count("n_neg_gate_pass")
                self._dense_update(j, ex, c, -1, fires)

    def _dense_update(self, j: int, ex: _Example, c: int, delta: int, fires: bool) -> None:
        k = self._vote_slot(j, c)
        w = self.vote_w[j][k]
        type_i = (w >= 0) == (delta > 0)  # target: w >= 0 gets Type I; negative: w < 0 gets Type I
        if type_i and fires:
            self._type_i_firing_dense(j, ex)
        elif type_i:
            self._type_ib(j, ex)
        elif fires:
            self._type_ii_dense(j, ex)
        if fires:  # votes change only when the clause fires; they may pass through 0
            self.vote_w[j][k] = max(-W_MAX, min(W_MAX, w + delta))

    def _type_i_firing_dense(self, j: int, ex: _Example) -> None:
        """5.2.1 without insertion (and without budget, see _state_up)."""
        self._count("n_type1")
        ids = self.lit_id[j]
        for k in range(self.l_slots):
            if ids[k] == LIT_EMPTY:
                continue
            if self._lit_true(ex, ids[k]):
                self._state_up(j, k)
            elif self._bern(self.derived.inv_s_q32, Stream.TYPE1_DEC, ex.a, j, k):
                self._state_down(j, k, 1)

    def _type_ii_dense(self, j: int, ex: _Example) -> None:
        """5.2.2 without negated insertion."""
        self._count("n_type2")
        ids, states = self.lit_id[j], self.lit_state[j]
        for k in range(self.l_slots):
            if ids[k] != LIT_EMPTY and states[k] < INCLUDE_THRESHOLD and not self._lit_true(ex, ids[k]):
                self._state_up(j, k)

    # ------------------------------------------------------------------ introspection (tests, debugging)
    def check_invariants(self) -> list[str]:
        """Violations of the invariants of spec 2 plus consistency of the derived structures."""
        bad: list[str] = []
        postings: dict[int, list[int]] = {}
        for j in range(self.n_clauses):
            ids, states = self.lit_id[j], self.lit_state[j]
            cls, w = self.vote_cls[j], self.vote_w[j]
            used = [lit for lit in ids if lit != LIT_EMPTY]
            if len(set(used)) != len(used):
                bad.append(f"clause {j}: duplicate literal id")
            for k in range(self.l_slots):
                if (ids[k] == LIT_EMPTY) != (states[k] == 0):
                    bad.append(f"clause {j} literal slot {k}: not exactly cleared or state 0 in use")
            used_cls = [c for c in cls if c != CLS_EMPTY]
            if len(set(used_cls)) != len(used_cls):
                bad.append(f"clause {j}: duplicate class id")
            for k in range(self.v_slots):
                if cls[k] == CLS_EMPTY and w[k] != 0:
                    bad.append(f"clause {j} vote slot {k}: free slot with w != 0")
                if cls[k] != CLS_EMPTY and w[k] == 0 and not self.is_dense:
                    bad.append(f"clause {j} vote slot {k}: used slot with w == 0")
                if not -W_MAX <= w[k] <= W_MAX:
                    bad.append(f"clause {j} vote slot {k}: weight out of range")
            n_inc = sum(1 for s in states if s >= INCLUDE_THRESHOLD)
            if not self.is_dense and n_inc > self.cfg.budget:
                bad.append(f"clause {j}: n_inc {n_inc} > BUDGET")
            if n_inc != self.n_inc[j] and self.alive[j]:
                bad.append(f"clause {j}: derived n_inc {self.n_inc[j]} != {n_inc}")
            if not self.alive[j]:
                cleared = (not used and not used_cls and not any(states) and not any(w)
                           and self.decay_mark[j] == 0 and self.own_fire_mass[j] == 0
                           and self.stat_fire[j] == 0 and self.stat_tp[j] == 0
                           and self.birth_step[j] == 0 and self.last_tp_step[j] == 0)
                if not cleared:
                    bad.append(f"clause {j}: free but not in the cleared initial form")
            if self.anchor[j] != ANCHOR_NONE:
                postings.setdefault(self.anchor[j], []).append(j)
        if postings != self.index:
            bad.append("inverted index differs from the anchors")
        if sorted(self.free) != [j for j in range(self.n_clauses) if not self.alive[j]]:
            bad.append("free list differs from alive == 0")
        return bad

    def included_literals(self, j: int) -> list[int]:
        """Included literal ids of clause j read from the slot arrays (live state, any mode)."""
        ids, states = self.lit_id[j], self.lit_state[j]
        return sorted(ids[k] for k in range(self.l_slots) if ids[k] != LIT_EMPTY and states[k] >= INCLUDE_THRESHOLD)

    def clause_text(self, j: int, token_text=None) -> str:
        """Readable form of clause j: included literals (live state) and votes."""
        lits = [lit_text(lit, self.cfg.vocab, token_text, tabular=self.is_tabular) for lit in self.included_literals(j)]
        votes = [f"{c}:{w:+d}" for c, w in zip(self.vote_cls[j], self.vote_w[j]) if c != CLS_EMPTY]
        return f"#{j} IF {' AND '.join(lits) or 'TRUE'} THEN {' '.join(votes)}"
