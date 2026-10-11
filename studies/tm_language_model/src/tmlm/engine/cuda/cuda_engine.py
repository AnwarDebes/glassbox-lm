"""CudaEngine: docs/11 on the one V100 behind the EngineBase API (docs/03 "Implementations"; docs/04 M4).

Every learning decision is made by the routines of core/tm_core.h, compiled by nvcc into
kernels.cu (kernel map K1..K14 in that file's header). This wrapper owns the device arrays in
the canonical STATE_FIELDS layout, builds the parameter, corpus and state structs with the same
ctypes definitions the CPU engine uses (the device layout is checked against `k_sizes` at
start), and runs the batch of docs/11 section 5 in the normative order:

    K1 examples -> K2 lookups -> K3 expand + verify -> K4 compaction (CuPy nonzero)
    -> K5 scores (int32 atomics) -> K6 gates, cover, negatives, seed requests
    -> K7/K8 events sorted by (clause, example) (CuPy sort on unique keys)
    -> K9 feedback, one thread per touched clause, events in ascending i
    -> K10 gate mass -> K11 seeding (literal draws parallel, serving on one thread)
    -> K12 refresh of changed clauses + index rebuild (CuPy sort of (anchor, clause) keys)
    -> K13 sweep every SWEEP_PERIOD batches -> step += 1

Bit-exactness with the CPU engine is asserted by tests/test_cuda_engine_parity.py on
`state_hash()` and on every DIAG counter. Nothing here depends on the block size, on the grid
size (all kernels are grid-stride loops) or on the batch composition beyond what the spec
defines: the only concurrent writes are commutative integer atomics.

Randomness: counter-based only (tm_rng64 in tm_core.h), no curand. The "RNG position" of a run
is therefore the pair (epoch, batch_index) plus the `step` counter inside the state; a
checkpoint (tmlm.engine.checkpoint) stores exactly that.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..api import DIAG_COUNTERS, STATE_FIELDS, CorpusArrays, EngineBase, TabularData, state_shapes
from ..config import ANCHOR_NONE, CLS_EMPTY, LIT_EMPTY, EngineConfig
from ..cpu.cpu_engine import _Corpus, _Params, occurrence_lists
from ..cpu.cpu_engine import CpuEngine as _CpuEngine
from . import build as _build

U32_MAX = 0xFFFFFFFF
PHASES = ("build", "lookup", "verify", "scores", "select", "events", "feedback", "gate_mass", "seeding",
          "refresh", "index", "sweep")
KERNELS = ("k_sizes", "k1_perm_ids", "k1_build_examples", "k2_count_candidates", "k3_expand", "k3_verify",
           "k_fired_bruteforce", "k5_scores", "k6_cover", "k6_select", "k9_feedback", "k9_feedback_eager", "k9_dense_feedback",
           "k10_gate_mass", "k11a_seed_literals", "k11b_seed_serve", "k11c_seed_serve_assigned", "k12_refresh", "k13_sweep", "k13_recycle_list",
           "k13_halve_stats", "k_init_dense")


class _State(C.Structure):
    """Mirror of tm_state (tm_core.h): 12 state pointers, step, 4 derived pointers."""

    _fields_ = [(name, C.c_void_p) for name, _, _ in STATE_FIELDS[:-1]] + [("step", C.c_uint64)] + [
        ("inc_lit", C.c_void_p), ("n_inc", C.c_void_p), ("n_inc_neg", C.c_void_p), ("anchor", C.c_void_p)]


_module_cache: dict[bool, Any] = {}


def load_module(debug: bool = False):
    """cupy.RawModule over the sm_70 cubin (built by build.py when stale)."""
    import cupy as cp
    if debug in _module_cache:
        return _module_cache[debug]
    if _build.needs_build(debug):
        _build.build(debug=debug)
    mod = cp.RawModule(path=str(_build.cubin_path(debug)))
    _module_cache[debug] = mod
    return mod


@dataclass
class _Fwd:
    n: int
    ex: Any            # device tm_example[n] as uint8 bytes
    fired_ex: Any      # u32 [n_fired], example-major
    fired_j: Any       # u32 [n_fired]
    flags: Any         # dense mode: u8 [C] fires per clause, else None
    scores: Any        # i32 [n, V]
    n_candidates: int


class CudaEngine(EngineBase):
    """docs/11 on the GPU. Bit-exact with the CPU engine: see tests/test_cuda_engine_parity.py.

    block: threads per block of every kernel launch (the result does not depend on it).
    max_grid: cap on the number of blocks (grid-stride loops cover the rest).
    predict_chunk: examples per forward pass in predict_scores (bounds the [chunk, V] score rows).
    profile: accumulate per-phase device time (CUDA events) into `self.profile_ms`, and count
    the included literals of verified candidates in `self.profile_counts` (experiment CM).
    """

    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays | None = None, tabular: TabularData | None = None,
                 block: int = 128, max_grid: int = 4096, predict_chunk: int = 4096, profile: bool = False,
                 debug: bool = False, own_range: tuple[int, int] | None = None) -> None:
        super().__init__(cfg, corpus, tabular)
        # docs/11 section 9: a shard OWNS the contiguous id range [lo, hi); its free list holds only own ids,
        # so foreign clauses are never seeded (and, never alive, never touched). None = the whole model.
        self.own_range = (0, cfg.n_clauses) if own_range is None else (int(own_range[0]), int(own_range[1]))
        if not 0 <= self.own_range[0] < self.own_range[1] <= cfg.n_clauses:
            raise ValueError("own_range must be a non-empty sub-range of [0, n_clauses)")
        if cfg.l_slots > 64 or cfg.v_slots > 1024 or cfg.k_neg > 8:
            raise ValueError("the CUDA engine supports l_slots <= 64, v_slots <= 1024, k_neg <= 8")
        if block < 1 or block > 1024 or max_grid < 1:
            raise ValueError("block must be in 1..1024 and max_grid >= 1")
        import cupy as cp
        self.cp = cp
        self.block, self.max_grid, self.predict_chunk = int(block), int(max_grid), int(predict_chunk)
        self.profile = bool(profile)
        self.profile_ms: dict[str, float] = {p: 0.0 for p in PHASES}
        self.profile_counts: dict[str, int] = {"n_inc_of_candidates": 0, "batches": 0, "sweeps": 0}
        self.mod = load_module(debug)
        self.kern = {name: self.mod.get_function(name) for name in KERNELS}
        self.is_dense = bool(cfg.dense_equivalent)
        self.is_tabular = cfg.mode == "tabular"
        self.step = 0
        self._check_layout()
        # parameters (host-computed integers, identical to the CPU engine's struct)
        self.params = _CpuEngine._make_params(self)  # type: ignore[arg-type]
        self._P = self._upload_struct(self.params)
        self._keep: list[Any] = []
        self._X = self._upload_struct(self._make_corpus())
        self._alloc_state()
        self._Sbuf = cp.zeros(C.sizeof(_State), dtype=cp.uint8)
        self._sync_state_struct()
        if self.is_dense:
            self._launch("k_init_dense", cfg.n_clauses, self._P, self._S)
        self._rebuild_derived()
        self._marks: list[tuple[str, Any]] = []

    # ---------------------------------------------------------------- setup
    def _check_layout(self) -> None:
        cp = self.cp
        out = cp.zeros(4, dtype=cp.uint64)
        self.kern["k_sizes"]((1,), (1,), (out,))
        dev = [int(v) for v in out.get()]
        host = [C.sizeof(_Params), C.sizeof(_Corpus), C.sizeof(_State)]
        if dev[:3] != host:
            raise RuntimeError(f"device struct sizes {dev[:3]} differ from the host ctypes sizes {host}")
        self.ex_size = dev[3]

    def _upload_struct(self, s: C.Structure):
        buf = np.frombuffer(bytes(s), dtype=np.uint8).copy()
        return self.cp.asarray(buf)

    def _dev(self, arr: np.ndarray, dtype):
        d = self.cp.asarray(np.ascontiguousarray(arr, dtype=dtype))
        self._keep.append(d)
        return d

    def _make_corpus(self) -> _Corpus:
        c = _Corpus()
        if self.tabular is not None:
            x = self._dev(self.tabular.x, np.uint8)
            y = self._dev(self.tabular.y, np.uint16)
            c.x, c.y, c.n_rows = x.data.ptr, y.data.ptr, int(y.shape[0])
            return c
        corpus = self.corpus
        assert corpus is not None
        occ_ptr, occ_pos = occurrence_lists(corpus, self.cfg.vocab, self.cfg.n_pol)
        tok = self._dev(corpus.tok, np.uint16)
        doc_start = self._dev(corpus.doc_start, np.uint32)
        kind_map = self._dev(corpus.kind_map, np.uint16)
        stop = self._dev(corpus.stop, np.uint8)
        sym_freq = self._dev(corpus.sym_freq, np.uint32)
        targets = self._dev(corpus.targets, np.uint32)
        d_occ_ptr, d_occ_pos = self._dev(occ_ptr, np.uint32), self._dev(occ_pos, np.uint32)
        self._n_tok = int(tok.shape[0])
        c.tok, c.n_tok = tok.data.ptr, int(tok.shape[0])
        c.doc_start, c.n_doc = doc_start.data.ptr, int(doc_start.shape[0]) - 1
        c.kind_map, c.stop, c.sym_freq = kind_map.data.ptr, stop.data.ptr, sym_freq.data.ptr
        c.targets, c.n_targets = targets.data.ptr, int(targets.shape[0])
        c.occ_ptr, c.occ_pos = d_occ_ptr.data.ptr, d_occ_pos.data.ptr
        return c

    def _alloc_state(self) -> None:
        cp, cfg = self.cp, self.cfg
        shapes = state_shapes(cfg)
        self.d: dict[str, Any] = {}
        for name, dtype, _ in STATE_FIELDS[:-1]:
            self.d[name] = cp.zeros(shapes[name], dtype=dtype)
        self.d["lit_id"][:] = LIT_EMPTY
        self.d["vote_cls"][:] = CLS_EMPTY
        self.d["inc_lit"] = cp.zeros((cfg.n_clauses, cfg.l_slots), dtype=cp.uint32)
        self.d["n_inc"] = cp.zeros(cfg.n_clauses, dtype=cp.uint8)
        self.d["n_inc_neg"] = cp.zeros(cfg.n_clauses, dtype=cp.uint8)
        self.d["anchor"] = cp.full(cfg.n_clauses, ANCHOR_NONE, dtype=cp.uint32)
        self._changed = cp.zeros(cfg.n_clauses, dtype=cp.uint8)
        self._anchor_changed = cp.zeros(1, dtype=cp.uint32)
        self._n_inc_sum = cp.zeros(1, dtype=cp.uint64)
        self._free_ids = cp.zeros(0, dtype=cp.uint32)
        self._free_head = cp.zeros(1, dtype=cp.uint32)
        self._idx_keys = cp.zeros(0, dtype=cp.uint32)
        self._idx_start = cp.zeros(1, dtype=cp.uint32)
        self._idx_clause = cp.zeros(0, dtype=cp.uint32)

    def _sync_state_struct(self) -> None:
        s = _State()
        for name, _, _ in STATE_FIELDS[:-1]:
            setattr(s, name, self.d[name].data.ptr)
        s.step = self.step
        for name in ("inc_lit", "n_inc", "n_inc_neg", "anchor"):
            setattr(s, name, self.d[name].data.ptr)
        self._Sbuf.set(np.frombuffer(bytes(s), dtype=np.uint8))
        self._S = self._Sbuf

    # ---------------------------------------------------------------- launches and profiling
    def _launch(self, name: str, n: int, *args: Any) -> None:
        if n <= 0:
            return
        grid = max(1, min(self.max_grid, (int(n) + self.block - 1) // self.block))
        self.kern[name]((grid,), (self.block,), tuple(args))

    def _mark(self, phase: str) -> None:
        if self.profile:
            ev = self.cp.cuda.Event()
            ev.record()
            self._marks.append((phase, ev))

    def _flush_marks(self) -> None:
        if not self.profile or not self._marks:
            return
        self.cp.cuda.Device().synchronize()
        for (_, prev), (phase, ev) in zip(self._marks, self._marks[1:]):
            self.profile_ms[phase] += float(self.cp.cuda.get_elapsed_time(prev, ev))
        self._marks = []

    @staticmethod
    def _u32(v: int) -> np.uint32:
        return np.uint32(v)

    _NULL = np.uint64(0)

    # ---------------------------------------------------------------- derived structures
    def _rebuild_index(self) -> None:
        """Anchor inverted index: unique anchor literals ascending, CSR into clause ids ascending."""
        cp = self.cp
        alive, anchor = self.d["alive"], self.d["anchor"]
        js = cp.flatnonzero((alive != 0) & (anchor != ANCHOR_NONE))
        if js.size == 0:
            self._idx_keys = cp.zeros(0, dtype=cp.uint32)
            self._idx_start = cp.zeros(1, dtype=cp.uint32)
            self._idx_clause = cp.zeros(0, dtype=cp.uint32)
            return
        keys = (anchor[js].astype(cp.uint64) << 32) | js.astype(cp.uint64)
        keys.sort()
        lits = (keys >> 32).astype(cp.uint32)
        self._idx_clause = (keys & U32_MAX).astype(cp.uint32)
        first = cp.concatenate([cp.ones(1, dtype=cp.bool_), lits[1:] != lits[:-1]])
        starts = cp.flatnonzero(first)
        self._idx_keys = lits[starts]
        self._idx_start = cp.concatenate([starts, cp.asarray([int(lits.size)])]).astype(cp.uint32)
        self._anchor_changed[:] = 0

    def _rebuild_free_list(self) -> None:
        cp = self.cp
        lo, hi = self.own_range
        self._free_ids = (cp.flatnonzero(self.d["alive"][lo:hi] == 0) + lo).astype(cp.uint32)
        self._free_head[:] = 0

    def _rebuild_derived(self) -> None:
        self._launch("k12_refresh", self.cfg.n_clauses, self._P, self._X, self._S, self._changed, np.int32(1), self._anchor_changed)
        self._rebuild_index()
        self._rebuild_free_list()

    def n_free(self) -> int:
        return int(self._free_ids.size) - int(self._free_head.get()[0])

    # ---------------------------------------------------------------- forward pass (spec 4)
    def _forward(self, epoch: int, ids_dev, bitmap_dev, training: int) -> _Fwd:
        cp, cfg, P = self.cp, self.cfg, self._P
        n = int(ids_dev.shape[0])
        ex = cp.empty(n * self.ex_size, dtype=cp.uint8)
        self._mark("start")
        self._launch("k1_build_examples", n, P, self._X, ex, self._u32(epoch), ids_dev, self._u32(n),
                     bitmap_dev if bitmap_dev is not None else self._NULL)
        self._mark("build")
        scores = cp.zeros((n, cfg.vocab), dtype=cp.int32)
        if self.is_dense:
            flags = cp.empty(n * cfg.n_clauses, dtype=cp.uint8)
            self._launch("k_fired_bruteforce", n * cfg.n_clauses, P, self._X, self._S, ex, self._u32(n), np.int32(1), np.int32(training), flags)
            self._mark("lookup")
            sel = cp.flatnonzero(flags)
            fired_ex = (sel // cfg.n_clauses).astype(cp.uint32)
            fired_j = (sel % cfg.n_clauses).astype(cp.uint32)
            n_cand = n * cfg.n_clauses
            self._mark("verify")
        else:
            flags = None
            n_slots = cfg.n_features if self.is_tabular else 4 * len(self.derived.exact_seg_ids)
            total_slots = n * n_slots
            slot_count = cp.empty(total_slots, dtype=cp.uint32)
            slot_first = cp.empty(total_slots, dtype=cp.uint32)
            self._launch("k2_count_candidates", total_slots, P, ex, self._u32(n), self._u32(n_slots),
                         self._idx_keys, self._idx_start, self._u32(int(self._idx_keys.size)), slot_count, slot_first)
            incl = cp.cumsum(slot_count, dtype=cp.uint64)
            n_cand = int(incl[-1].get()) if total_slots else 0
            slot_off = (incl - slot_count).astype(cp.uint32)
            self._mark("lookup")
            cand_ex = cp.empty(n_cand, dtype=cp.uint32)
            cand_j = cp.empty(n_cand, dtype=cp.uint32)
            self._launch("k3_expand", total_slots, self._u32(total_slots), self._u32(n_slots), slot_count, slot_first,
                         slot_off, self._idx_clause, cand_ex, cand_j)
            fired = cp.empty(n_cand, dtype=cp.uint8)
            self._launch("k3_verify", n_cand, P, self._X, self._S, ex, self._u32(n_cand), cand_ex, cand_j, fired, self._n_inc_sum)
            sel = cp.flatnonzero(fired)
            fired_ex, fired_j = cand_ex[sel], cand_j[sel]
            self._mark("verify")
        self._launch("k5_scores", int(fired_j.size), P, self._S, self._u32(int(fired_j.size)), fired_ex, fired_j, scores)
        self._mark("scores")
        return _Fwd(n, ex, fired_ex, fired_j, flags, scores, n_cand)

    # ---------------------------------------------------------------- training (spec 5), as phases
    # The phases mirror the oracle's phase methods (oracle.py: _forward_partial, _forward_finish,
    # _phase_feedback, _gate_mass, _seed_request, _seed_literals, _seed_serve, _phase_refresh, _sweep_local,
    # _sweep_candidates, _sweep_shift) so that cuda/shard.py can put the three exchanges of docs/11
    # section 9 between them. `_train` composes them for the unsharded engine.

    @dataclass
    class _Batch:
        epoch: int
        n: int
        diag: Any
        fw: Any = None
        fired_off: Any = None
        cover: Any = None       # u32 [n], this engine's (partial) cover counts
        seed_req: Any = None    # u8 [n]
        scores: Any = None      # i32 [n, V] the scores every later phase reads (summed in a sharded run)
        req_i: Any = None       # u32 [n_req] requesting examples, ascending
        lits: Any = None        # u32 [3 * n_req]
        n_lits: Any = None      # u32 [n_req]

    def _phase_forward(self, epoch: int, ids_dev) -> "CudaEngine._Batch":
        """Spec 4 steps 1 to 4 over this engine's clauses: partial scores and partial cover (oracle _forward_partial)."""
        cp, cfg, P = self.cp, self.cfg, self._P
        self._sync_state_struct()
        n = int(ids_dev.shape[0])
        if self.is_dense and n != 1:
            raise ValueError("dense-equivalent mode needs batches of exactly one example (N9)")
        bt = self._Batch(epoch=epoch, n=n, diag=cp.zeros(len(DIAG_COUNTERS), dtype=cp.uint64))
        bt.fw = self._forward(epoch, ids_dev, None, 1)
        n_fired = int(bt.fw.fired_j.size)
        counts = cp.bincount(bt.fw.fired_ex, minlength=n) if n_fired else cp.zeros(n, dtype=cp.int64)
        bt.fired_off = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)]).astype(cp.uint32)
        bt.cover = cp.zeros(n, dtype=cp.uint32)
        self._launch("k6_cover", n, P, self._S, bt.fw.ex, self._u32(n), bt.fired_off, bt.fw.fired_j, bt.cover)
        return bt

    def _phase_select(self, bt: "CudaEngine._Batch", scores, cover) -> None:
        """Spec 4 step 5 and the seed request of 5.6 from the (summed) scores and cover (oracle _forward_finish,
        _seed_request). `scores` is the i32 [n, V] matrix every later phase reads."""
        bt.scores = scores
        bt.seed_req = self.cp.zeros(bt.n, dtype=self.cp.uint8)
        self._launch("k6_select", bt.n, self._P, bt.fw.ex, self._u32(bt.n), cover, scores, bt.seed_req)
        self._mark("select")

    def _phase_feedback(self, bt: "CudaEngine._Batch") -> None:
        """Spec 5.1 to 5.4 on this engine's clauses (oracle _phase_feedback); dense mode: spec 6."""
        cp, cfg, P, X, S = self.cp, self.cfg, self._P, self._X, self._S
        fw, n, scores, diag = bt.fw, bt.n, bt.scores, bt.diag
        if self.is_dense:
            self._launch("k9_dense_feedback", cfg.n_clauses, P, X, S, fw.ex, fw.flags, scores, diag)
            self._mark("feedback")
            return
        n_fired = int(fw.fired_j.size)
        # K7, K8: events grouped by clause, ascending example index inside a clause
        keys = (fw.fired_j.astype(cp.uint64) << 32) | fw.fired_ex.astype(cp.uint64)
        keys.sort()
        ev_j = (keys >> 32).astype(cp.uint32)
        ev_i = (keys & U32_MAX).astype(cp.uint32)
        if n_fired:
            first = cp.concatenate([cp.ones(1, dtype=cp.bool_), ev_j[1:] != ev_j[:-1]])
            starts = cp.flatnonzero(first)
            touched_j = ev_j[starts]
            t_start = starts.astype(cp.uint32)
            t_end = cp.concatenate([starts[1:], cp.asarray([n_fired])]).astype(cp.uint32)
        else:
            touched_j = t_start = t_end = cp.zeros(0, dtype=cp.uint32)
        n_touched = int(touched_j.size)
        self._mark("events")
        if cfg.decay_mode == "lazy_owned":
            self._launch("k9_feedback", n_touched, P, X, S, fw.ex, self._u32(n_touched), touched_j, t_start, t_end,
                         ev_i, scores, self._changed, diag)
        else:  # N10
            ev_ptr = cp.full(cfg.n_clauses, U32_MAX, dtype=cp.uint32)
            ev_cnt = cp.zeros(cfg.n_clauses, dtype=cp.uint32)
            if n_touched:
                ev_ptr[touched_j] = t_start
                ev_cnt[touched_j] = t_end - t_start
            self._launch("k9_feedback_eager", cfg.n_clauses, P, X, S, fw.ex, self._u32(n), ev_ptr, ev_cnt, ev_i,
                         scores, self._changed, diag)
        self._mark("feedback")

    def _phase_gate_mass(self, bt: "CudaEngine._Batch") -> None:
        """Spec 5.5 (oracle _gate_mass); a pure function of the examples, identical on every shard."""
        self._launch("k10_gate_mass", bt.n, self._S, bt.fw.ex, self._u32(bt.n))
        self._mark("gate_mass")

    def _phase_seed_literals(self, bt: "CudaEngine._Batch") -> int:
        """Requests in ascending i and their literal draws (oracle _seed_literals: a function of the example
        only). Returns the number of requests; `bt.req_i`, `bt.lits`, `bt.n_lits` are set."""
        cp, P, X = self.cp, self._P, self._X
        bt.req_i = cp.flatnonzero(bt.seed_req).astype(cp.uint32)
        n_req = int(bt.req_i.size)
        bt.lits = cp.zeros(3 * n_req, dtype=cp.uint32)
        bt.n_lits = cp.zeros(n_req, dtype=cp.uint32)
        if n_req:
            self._launch("k11a_seed_literals", n_req, P, X, bt.fw.ex, self._u32(n_req), bt.req_i, bt.lits, bt.n_lits)
        return n_req

    def _phase_seed_serve(self, bt: "CudaEngine._Batch") -> None:
        """Spec 5.6 serving on one thread from this engine's free list (oracle _seed_serve with the cap and the
        request count): the unsharded path."""
        cp, cfg = self.cp, self.cfg
        n_req = int(bt.req_i.size)
        if n_req:
            served = cp.zeros(cfg.vocab, dtype=cp.uint32)
            self.kern["k11b_seed_serve"]((1,), (1,), (self._P, self._S, bt.fw.ex, self._u32(n_req), bt.req_i, bt.lits, bt.n_lits,
                                                      served, self._free_ids, self._u32(int(self._free_ids.size)),
                                                      self._free_head, self._changed, bt.diag))
        self._mark("seeding")

    def _seed_serve_assigned(self, bt: "CudaEngine._Batch", which) -> None:
        """Sharded serving (docs/11 section 9, exchange 2): `which` = indices into bt.req_i of the requests this
        shard serves, ascending, each with at least one literal and a free clause reserved by the caller."""
        cp = self.cp
        idx = cp.asarray(np.asarray(which, dtype=np.int64))
        n = int(idx.size)
        if n:
            req_i = bt.req_i[idx]
            lits = bt.lits.reshape(-1, 3)[idx].reshape(-1)
            n_lits = bt.n_lits[idx]
            self.kern["k11c_seed_serve_assigned"]((1,), (1,), (self._P, self._S, bt.fw.ex, self._u32(n), req_i, lits, n_lits,
                                                               self._free_ids, self._u32(int(self._free_ids.size)),
                                                               self._free_head, self._changed))
        self._mark("seeding")

    def _phase_refresh(self) -> None:
        """Refresh of changed clauses, then the index if any anchor moved (oracle _phase_refresh)."""
        self._launch("k12_refresh", self.cfg.n_clauses, self._P, self._X, self._S, self._changed, np.int32(0), self._anchor_changed)
        self._mark("refresh")
        if int(self._anchor_changed.get()[0]):
            self._rebuild_index()
        self._mark("index")

    def sweep_due(self) -> bool:
        return (self.step + 1) % self.cfg.sweep_period == 0  # N1

    def _sweep_local(self, diag) -> None:
        """Spec 5.7 per clause: settle, refresh, recycle by anchor, ownership and age (oracle _sweep_local)."""
        self._launch("k13_sweep", self.cfg.n_clauses, self._P, self._X, self._S, diag)

    def n_free_now(self) -> int:
        """Free clauses of the owned range from `alive` (valid between _sweep_local and _sweep_shift)."""
        lo, hi = self.own_range
        return (hi - lo) - int(self.d["alive"][lo:hi].sum().get())

    def _sweep_candidates(self, limit: int) -> np.ndarray:
        """The top-up candidates of spec 5.7: alive clauses older than 2 * SWEEP_PERIOD, ascending (stat_tp, id),
        truncated to `limit` (oracle _sweep_candidates). Returns u64 keys (stat_tp << 32 | id) on the host."""
        cp, cfg = self.cp, self.cfg
        step_lo = self.step & U32_MAX
        age = (step_lo - self.d["birth_step"].astype(cp.int64)) & U32_MAX
        js = cp.flatnonzero((self.d["alive"] != 0) & (age > 2 * cfg.sweep_period))
        if js.size == 0:
            return np.zeros(0, dtype=np.uint64)
        keys = (self.d["stat_tp"][js].astype(cp.uint64) << 32) | js.astype(cp.uint64)
        keys.sort()
        return keys[:limit].get()

    def _recycle_list(self, ids: np.ndarray, diag) -> None:
        ids = np.asarray(ids, dtype=np.uint32)
        if ids.size:
            d_ids = self.cp.asarray(ids)
            self._launch("k13_recycle_list", int(ids.size), self._P, self._X, self._S, self._u32(int(ids.size)), d_ids, diag)

    def _sweep_shift(self) -> None:
        """Halve the fire statistics, rebuild the index and the free list (oracle _sweep_shift)."""
        self._launch("k13_halve_stats", self.cfg.n_clauses, self._P, self._S)
        self._rebuild_index()
        self._rebuild_free_list()
        if self.profile:
            self.profile_counts["sweeps"] += 1

    def _sweep(self, diag) -> None:  # spec 5.7, unsharded
        self._sweep_local(diag)
        need = self.cfg.n_clauses // 128 - self.n_free_now()
        if need > 0:
            keys = self._sweep_candidates(need)
            self._recycle_list((keys & U32_MAX).astype(np.uint32), diag)
        self._sweep_shift()

    def _finish_batch(self, bt: "CudaEngine._Batch") -> dict[str, int]:
        """step += 1, flush the profile, return this engine's DIAG counters (n_examples, n_candidates and
        n_fired added here from the forward pass)."""
        self.step += 1
        self._flush_marks()
        d = {name: int(v) for name, v in zip(DIAG_COUNTERS, bt.diag.get())}
        d["n_examples"] += bt.n
        d["n_candidates"] += bt.fw.n_candidates
        d["n_fired"] += int(bt.fw.fired_j.size)
        if self.profile:
            self.profile_counts["batches"] += 1
            self.profile_counts["n_inc_of_candidates"] = int(self._n_inc_sum.get()[0])
        return d

    def _train(self, epoch: int, ids_dev) -> dict[str, int]:
        bt = self._phase_forward(epoch, ids_dev)
        self._phase_select(bt, bt.fw.scores, bt.cover)
        self._phase_feedback(bt)
        self._phase_gate_mass(bt)
        if not self.is_dense:
            self._phase_seed_literals(bt)
            self._phase_seed_serve(bt)
            self._phase_refresh()
            if self.sweep_due():
                self._sweep(bt.diag)
                self._mark("sweep")
        return self._finish_batch(bt)

    # ---------------------------------------------------------------- EngineBase API
    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        cp = self.cp
        n_total = self.n_examples
        lo = int(batch_index) * self.cfg.batch
        if lo >= n_total or batch_index < 0:
            raise ValueError("batch_index out of range")
        hi = min(n_total, lo + self.cfg.batch)
        ids = cp.empty(hi - lo, dtype=cp.uint32)
        self._launch("k1_perm_ids", hi - lo, self._P, self._X, self._u32(epoch), np.uint64(lo), self._u32(hi - lo),
                     np.uint64(n_total), ids)
        return self._train(epoch, ids)

    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        ids = np.ascontiguousarray(np.asarray(examples).reshape(-1), dtype=np.uint32)
        return self._train(epoch, self.cp.asarray(ids))

    def _bitmap(self, hidden: np.ndarray | None):
        if hidden is None or self.cfg.mode not in ("masked", "diffusion"):
            return None
        b = np.ascontiguousarray(hidden, dtype=np.uint8)
        if b.shape != (self._n_tok,):
            raise ValueError("hidden must be a bitmap over all token positions")
        return self.cp.asarray(b)

    def predict_scores(self, epoch: int, examples: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
        cp = self.cp
        self._sync_state_struct()
        ids = np.ascontiguousarray(np.asarray(examples).reshape(-1), dtype=np.uint32)
        out = np.zeros((ids.shape[0], self.cfg.vocab), dtype=np.int32)
        bitmap = self._bitmap(hidden)
        for lo in range(0, ids.shape[0], self.predict_chunk):
            chunk = cp.asarray(ids[lo:lo + self.predict_chunk])
            fw = self._forward(epoch, chunk, bitmap, 0)
            out[lo:lo + self.predict_chunk] = fw.scores.get()
        self._flush_marks()
        return out

    def fired_clauses(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        cp = self.cp
        self._sync_state_struct()
        fw = self._forward(epoch, cp.asarray(np.array([example], dtype=np.uint32)), self._bitmap(hidden), 0)
        self._marks = []
        return np.sort(fw.fired_j.get().astype(np.uint32))

    def fired_clauses_bruteforce(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        cp = self.cp
        self._sync_state_struct()
        ex = cp.empty(self.ex_size, dtype=cp.uint8)
        bitmap = self._bitmap(hidden)
        self._launch("k1_build_examples", 1, self._P, self._X, ex, self._u32(epoch), cp.asarray(np.array([example], dtype=np.uint32)),
                     self._u32(1), bitmap if bitmap is not None else self._NULL)
        flags = cp.empty(self.cfg.n_clauses, dtype=cp.uint8)
        self._launch("k_fired_bruteforce", self.cfg.n_clauses, self._P, self._X, self._S, ex, self._u32(1), np.int32(int(self.is_dense)), np.int32(0), flags)
        return cp.flatnonzero(flags).get().astype(np.uint32)

    def state(self) -> dict[str, np.ndarray]:
        st = {name: self.d[name].get() for name, _, _ in STATE_FIELDS[:-1]}
        st["step"] = np.array([self.step], dtype=np.uint64)
        return st

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        shapes = state_shapes(self.cfg)
        for name, dtype, _ in STATE_FIELDS[:-1]:
            arr = np.ascontiguousarray(state[name], dtype=dtype)
            if arr.shape != shapes[name]:
                raise ValueError(f"state[{name!r}]: expected shape {shapes[name]}")
            self.d[name].set(arr)
        self.step = int(state["step"][0])
        self._sync_state_struct()
        self._changed[:] = 0
        self._rebuild_derived()

    def synchronize(self) -> None:
        self.cp.cuda.Device().synchronize()

    def device_memory_used(self) -> int:
        """Bytes currently held by CuPy's pool plus the raw device allocations (peak memory proxy)."""
        pool = self.cp.get_default_memory_pool()
        return int(pool.used_bytes())

    def reset_profile(self) -> None:
        self.profile_ms = {p: 0.0 for p in PHASES}
        self.profile_counts = {"n_inc_of_candidates": 0, "batches": 0, "sweeps": 0}
        self._n_inc_sum[:] = 0
        self._marks = []
