"""CpuEngine: the C++/OpenMP host engine behind the EngineBase API (docs/11; docs/04 M3).

The C library (engine.cpp + core/tm_core.h) holds all learning logic. This wrapper builds the
parameter struct from `EngineConfig` and `derive()`, keeps the corpus arrays (and the occurrence
lists needed for bag literals) alive for the lifetime of the engine, and moves state in and out
in the canonical `STATE_FIELDS` layout so that `state_hash()` is comparable with the oracle.
"""

from __future__ import annotations

import ctypes as C
from fractions import Fraction

import numpy as np

from ..api import DIAG_COUNTERS, STATE_FIELDS, CorpusArrays, EngineBase, TabularData, state_shapes
from ..config import (
    NEG_K_MAX,
    MAX_BAG_K,
    MODES,
    N_EXACT_SEG,
    DECAY_MODES,
    NEG_MODES,
    EngineConfig,
    n_sym,
    sym_edge,
    sym_mask,
)
from . import build as _build

_MAX_SEG = N_EXACT_SEG + 2 * MAX_BAG_K


class _Params(C.Structure):
    _fields_ = [
        ("C", C.c_uint32), ("V", C.c_uint32), ("L_SLOTS", C.c_uint32), ("V_SLOTS", C.c_uint32),
        ("BUDGET", C.c_uint32), ("K_NEG", C.c_uint32), ("K_COVER", C.c_uint32), ("SEED_CAP", C.c_uint32),
        ("E_CAP", C.c_uint32), ("SWEEP_PERIOD", C.c_uint32), ("DEAD_SWEEPS", C.c_uint32),
        ("T", C.c_int32), ("MARGIN", C.c_int32),
        ("inv_s_q32", C.c_uint32), ("p_ins_q32", C.c_uint32), ("p_neg_q32", C.c_uint32), ("p_veto_q32", C.c_uint32),
        ("rho_seed_q32", C.c_uint32), ("mask_rate_q32", C.c_uint32), ("q_neg_q32", C.c_uint32),
        ("kind_t70", C.c_uint32), ("kind_t80", C.c_uint32), ("kind_t90", C.c_uint32), ("nlit_t1", C.c_uint32), ("nlit_t2", C.c_uint32),
        ("seed", C.c_uint64),
        ("s_num", C.c_uint64), ("s_den", C.c_uint64),
        ("mode", C.c_uint32), ("neg_mode", C.c_uint32), ("decay_mode", C.c_uint32), ("dense_equivalent", C.c_uint32),
        ("n_bag_k", C.c_uint32), ("n_features", C.c_uint32),
        ("sym_mask", C.c_uint32), ("sym_edge", C.c_uint32), ("n_sym", C.c_uint32),
        ("min_free", C.c_uint32),
        ("neg_budget", C.c_int32),
        ("cand_tries", C.c_uint32),
        ("enabled_segs", C.c_uint64),
        ("n_segs", C.c_uint32), ("seg_total", C.c_uint32), ("seg_ids", C.c_uint32 * _MAX_SEG), ("seg_cum", C.c_uint32 * _MAX_SEG),
        ("n_exact", C.c_uint32), ("exact_total", C.c_uint32), ("exact_seg_ids", C.c_uint32 * N_EXACT_SEG), ("exact_cum", C.c_uint32 * N_EXACT_SEG),
        ("neg_cdf_q32", C.c_uint32 * (NEG_K_MAX + 1)),
    ]


class _Corpus(C.Structure):
    _fields_ = [
        ("tok", C.c_void_p), ("n_tok", C.c_uint64),
        ("doc_start", C.c_void_p), ("n_doc", C.c_uint32),
        ("kind_map", C.c_void_p), ("stop", C.c_void_p), ("sym_freq", C.c_void_p),
        ("targets", C.c_void_p), ("n_targets", C.c_uint32),
        ("occ_ptr", C.c_void_p), ("occ_pos", C.c_void_p),
        ("x", C.c_void_p), ("y", C.c_void_p), ("n_rows", C.c_uint32),
    ]


_lib: C.CDLL | None = None
_lib_debug: bool | None = None


def load_library(debug: bool = False) -> C.CDLL:
    global _lib, _lib_debug
    if _lib is not None and _lib_debug == debug:
        return _lib
    if _build.needs_build(debug):
        _build.build(debug=debug)
    lib = C.CDLL(str(_build.lib_path(debug)))
    lib.tm_create.restype = C.c_void_p
    lib.tm_create.argtypes = [C.POINTER(_Params), C.POINTER(_Corpus)]
    lib.tm_destroy.argtypes = [C.c_void_p]
    lib.tm_get_state.argtypes = [C.c_void_p] + [C.c_void_p] * 12 + [C.POINTER(C.c_uint64)]
    lib.tm_load_state.argtypes = [C.c_void_p] + [C.c_void_p] * 12 + [C.c_uint64]
    lib.tm_train_batch_b.restype = C.c_int
    lib.tm_train_batch_b.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p]
    lib.tm_train_explicit.restype = C.c_int
    lib.tm_train_explicit.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p]
    lib.tm_predict_scores.restype = C.c_int
    lib.tm_predict_scores.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p]
    lib.tm_fired_clauses.restype = C.c_uint32
    lib.tm_fired_clauses.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p, C.c_void_p, C.c_uint32, C.c_int]
    lib.tm_n_free.restype = C.c_uint32
    lib.tm_n_free.argtypes = [C.c_void_p]
    lib.tm_omp_max_threads.restype = C.c_int
    _lib, _lib_debug = lib, debug
    return lib


def occurrence_lists(corpus: CorpusArrays, vocab: int, n_pol: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """CSR occurrence lists per symbol (all four kinds), stop tokens excluded, positions ascending.
    Semantically identical to the oracle's bucket scan (docs/11 section 1.4)."""
    tok = corpus.tok.astype(np.int64)
    keep = corpus.stop[tok] == 0
    pos = np.nonzero(keep)[0].astype(np.uint32)
    toks = tok[keep]
    syms = np.concatenate([corpus.kind_map[k][toks].astype(np.int64) for k in range(4)])
    positions = np.concatenate([pos] * 4)
    order = np.lexsort((positions, syms))
    syms, positions = syms[order], positions[order]
    counts = np.bincount(syms, minlength=n_sym(vocab, n_pol))
    occ_ptr = np.zeros(n_sym(vocab, n_pol) + 1, dtype=np.uint32)
    occ_ptr[1:] = np.cumsum(counts)
    return occ_ptr, positions.astype(np.uint32)


def _ptr(arr: np.ndarray) -> int:
    return arr.ctypes.data


class CpuEngine(EngineBase):
    """docs/11 on the CPU. Bit-exact with the oracle: see tests/test_cpu_engine_parity.py."""

    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays | None = None,
                 tabular: TabularData | None = None, debug: bool = False) -> None:
        super().__init__(cfg, corpus, tabular)
        if cfg.l_slots > 64 or cfg.v_slots > 1024 or cfg.k_neg > 8:
            raise ValueError("the C engine supports l_slots <= 64, v_slots <= 1024, k_neg <= 8")
        self.lib = load_library(debug)
        self._keep: list[np.ndarray] = []
        self.params = self._make_params()
        self.cstruct = self._make_corpus()
        self.handle = self.lib.tm_create(C.byref(self.params), C.byref(self.cstruct))
        if not self.handle:
            raise RuntimeError("tm_create failed")

    def __del__(self) -> None:
        h = getattr(self, "handle", None)
        if h:
            self.lib.tm_destroy(h)
            self.handle = None

    # ---------------------------------------------------------------- parameters and corpus
    def _make_params(self) -> _Params:
        cfg, d = self.cfg, self.derived
        p = _Params()
        p.C, p.V, p.L_SLOTS, p.V_SLOTS = cfg.n_clauses, cfg.vocab, cfg.l_slots, cfg.v_slots
        p.BUDGET, p.K_NEG, p.K_COVER, p.SEED_CAP = cfg.budget, cfg.k_neg, cfg.k_cover, cfg.seed_cap
        p.E_CAP, p.SWEEP_PERIOD, p.DEAD_SWEEPS = cfg.e_cap, cfg.sweep_period, cfg.dead_sweeps
        p.T, p.MARGIN = cfg.T, d.margin
        p.inv_s_q32, p.p_ins_q32, p.p_neg_q32, p.p_veto_q32 = d.inv_s_q32, d.p_ins_q32, d.p_neg_q32, d.p_veto_q32
        p.rho_seed_q32, p.mask_rate_q32, p.q_neg_q32 = d.rho_seed_q32, d.mask_rate_q32, d.q_neg_q32
        p.kind_t70, p.kind_t80, p.kind_t90, p.nlit_t1, p.nlit_t2 = d.kind_t70, d.kind_t80, d.kind_t90, d.nlit_t1, d.nlit_t2
        p.seed = cfg.seed
        num, den = Fraction(cfg.s).as_integer_ratio()
        p.s_num, p.s_den = num, den
        p.mode, p.neg_mode, p.decay_mode = MODES.index(cfg.mode), NEG_MODES.index(cfg.neg_mode), DECAY_MODES.index(cfg.decay_mode)
        p.dense_equivalent = int(cfg.dense_equivalent)
        p.n_bag_k, p.n_features = cfg.n_bag_k, cfg.n_features
        p.sym_mask, p.sym_edge, p.n_sym = sym_mask(cfg.vocab), sym_edge(cfg.vocab), n_sym(cfg.vocab, cfg.n_pol)
        p.min_free = cfg.n_clauses // 128
        p.neg_budget = cfg.neg_budget
        p.cand_tries = cfg.cand_tries
        enabled = 0
        for seg in d.seg_ids:
            enabled |= 1 << seg
        p.enabled_segs = enabled
        p.n_segs, p.seg_total = len(d.seg_ids), d.seg_total
        for i, (s, c) in enumerate(zip(d.seg_ids, d.seg_cum)):
            p.seg_ids[i], p.seg_cum[i] = s, c
        p.n_exact, p.exact_total = len(d.exact_seg_ids), d.exact_total
        for i, (s, c) in enumerate(zip(d.exact_seg_ids, d.exact_cum)):
            p.exact_seg_ids[i], p.exact_cum[i] = s, c
        for i, v in enumerate(d.neg_cdf_q32):
            p.neg_cdf_q32[i] = v
        return p

    def _make_corpus(self) -> _Corpus:
        c = _Corpus()
        if self.tabular is not None:
            x = np.ascontiguousarray(self.tabular.x, dtype=np.uint8)
            y = np.ascontiguousarray(self.tabular.y, dtype=np.uint16)
            self._keep += [x, y]
            c.x, c.y, c.n_rows = _ptr(x), _ptr(y), int(y.shape[0])
            return c
        corpus = self.corpus
        assert corpus is not None
        tok = np.ascontiguousarray(corpus.tok, dtype=np.uint16)
        doc_start = np.ascontiguousarray(corpus.doc_start, dtype=np.uint32)
        kind_map = np.ascontiguousarray(corpus.kind_map, dtype=np.uint16)
        stop = np.ascontiguousarray(corpus.stop, dtype=np.uint8)
        sym_freq = np.ascontiguousarray(corpus.sym_freq, dtype=np.uint32)
        targets = np.ascontiguousarray(corpus.targets, dtype=np.uint32)
        occ_ptr, occ_pos_now = occurrence_lists(corpus, self.cfg.vocab, self.cfg.n_pol)
        # The occurrence buffer is allocated at its maximum size (every position in all 4 kinds) so
        # that `refresh_occurrences()` can rewrite it in place after the tokens of the bound corpus
        # change (generation windows, src/tmlm/decode/scorer.py); the C engine reads only the
        # prefix that occ_ptr addresses.
        occ_pos = np.zeros(4 * int(tok.shape[0]), dtype=np.uint32)
        occ_pos[: occ_pos_now.shape[0]] = occ_pos_now
        self._keep += [tok, doc_start, kind_map, stop, sym_freq, targets, occ_ptr, occ_pos]
        self.corpus_buffers = {"tok": tok, "doc_start": doc_start, "occ_ptr": occ_ptr, "occ_pos": occ_pos}
        c.tok, c.n_tok = _ptr(tok), int(tok.shape[0])
        c.doc_start, c.n_doc = _ptr(doc_start), int(doc_start.shape[0]) - 1
        c.kind_map, c.stop, c.sym_freq = _ptr(kind_map), _ptr(stop), _ptr(sym_freq)
        c.targets, c.n_targets = _ptr(targets), int(targets.shape[0])
        c.occ_ptr, c.occ_pos = _ptr(occ_ptr), _ptr(occ_pos)
        return c

    # ---------------------------------------------------------------- state
    def state(self) -> dict[str, np.ndarray]:
        shapes = state_shapes(self.cfg)
        st = {name: np.zeros(shapes[name], dtype=dtype) for name, dtype, _ in STATE_FIELDS}
        step = C.c_uint64(0)
        self.lib.tm_get_state(self.handle, *[_ptr(st[name]) for name, _, _ in STATE_FIELDS[:-1]], C.byref(step))
        st["step"][0] = step.value
        return st

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        shapes = state_shapes(self.cfg)
        arrs = []
        for name, dtype, _ in STATE_FIELDS[:-1]:
            arr = np.ascontiguousarray(state[name], dtype=dtype)
            if arr.shape != shapes[name]:
                raise ValueError(f"state[{name!r}]: expected shape {shapes[name]}")
            arrs.append(arr)
        self.lib.tm_load_state(self.handle, *[_ptr(a) for a in arrs], int(state["step"][0]))

    # ---------------------------------------------------------------- training and prediction
    def _diag(self, buf: np.ndarray) -> dict[str, int]:
        return {name: int(v) for name, v in zip(DIAG_COUNTERS, buf)}

    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        buf = np.zeros(len(DIAG_COUNTERS), dtype=np.uint64)
        rc = self.lib.tm_train_batch_b(self.handle, epoch, batch_index, self.cfg.batch, _ptr(buf))
        if rc != 0:
            raise ValueError("batch_index out of range")
        return self._diag(buf)

    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        ids = np.ascontiguousarray(np.asarray(examples).reshape(-1), dtype=np.uint32)
        buf = np.zeros(len(DIAG_COUNTERS), dtype=np.uint64)
        self.lib.tm_train_explicit(self.handle, epoch, _ptr(ids), int(ids.shape[0]), _ptr(buf))
        return self._diag(buf)

    def _bitmap(self, hidden: np.ndarray | None) -> np.ndarray | None:
        if hidden is None or self.cfg.mode not in ("masked", "diffusion"):
            return None
        b = np.ascontiguousarray(hidden, dtype=np.uint8)
        if b.shape != (int(self.corpus.tok.shape[0]),):  # type: ignore[union-attr]
            raise ValueError("hidden must be a bitmap over all token positions")
        return b

    def predict_scores(self, epoch: int, examples: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(examples).reshape(-1), dtype=np.uint32)
        out = np.zeros((ids.shape[0], self.cfg.vocab), dtype=np.int32)
        b = self._bitmap(hidden)
        self.lib.tm_predict_scores(self.handle, epoch, _ptr(ids), int(ids.shape[0]), _ptr(b) if b is not None else None, _ptr(out))
        return out

    def _fired(self, epoch: int, example: int, hidden: np.ndarray | None, bruteforce: bool) -> np.ndarray:
        b = self._bitmap(hidden)
        out = np.zeros(self.cfg.n_clauses, dtype=np.uint32)
        n = self.lib.tm_fired_clauses(self.handle, epoch, int(example), _ptr(b) if b is not None else None,
                                      _ptr(out), self.cfg.n_clauses, int(bruteforce))
        return out[:n].copy()

    def fired_clauses(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        return self._fired(epoch, example, hidden, False)

    def fired_clauses_bruteforce(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        return self._fired(epoch, example, hidden, True)

    def n_free(self) -> int:
        return int(self.lib.tm_n_free(self.handle))

    # ---------------------------------------------------------------- bound corpus buffers
    def refresh_occurrences(self) -> None:
        """Recompute the bag occurrence lists from the CURRENT contents of the bound `tok` buffer
        (`self.corpus_buffers["tok"]`, the very array the C engine reads). Call it after writing
        into that buffer; the exact-segment literals read `tok` directly and need no refresh. The
        static tables (kind_map, stop, sym_freq) and doc_start are not touched, so anchors and the
        inverted index stay valid. Language modes only."""
        if self.corpus is None:
            raise ValueError("refresh_occurrences needs a language corpus")
        bufs = self.corpus_buffers
        live = CorpusArrays(tok=bufs["tok"], doc_start=bufs["doc_start"], kind_map=self.corpus.kind_map,
                            stop=self.corpus.stop, sym_freq=self.corpus.sym_freq, targets=self.corpus.targets)
        occ_ptr, occ_pos = occurrence_lists(live, self.cfg.vocab, self.cfg.n_pol)
        bufs["occ_ptr"][:] = occ_ptr
        bufs["occ_pos"][: occ_pos.shape[0]] = occ_pos
