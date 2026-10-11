"""Clause sharding on the CUDA engine (docs/11 section 9; experiment s5, docs/05; docs/13 H3.5): the same
model trained as S `CudaEngine` instances on the one V100, shard r owning the contiguous clause id range
[r*C/S, (r+1)*C/S) (its free list holds only those ids, so foreign clauses are never seeded, never alive,
never touched). Between the phases of docs/11 section 5 this wrapper performs exactly the three
exchanges of section 9 and nothing else, mirroring `tmlm.engine.oracle.shard.ShardedOracle`:

  exchange 1  the summed score matrix S_i[c] (i32 [B, V], a CuPy sum of the partial matrices) and the
              summed cover counts, after the partial forward pass of every shard
  exchange 2  the free-clause count of every shard: seeding is served in ascending example index from the
              GLOBAL free list, lowest id first, with the per-class cap applied once on the host
  exchange 3  each shard's top-up candidate list (stat_tp, id) truncated to MIN_FREE, merged on the host

Everything else runs unchanged on the shards through the phase methods of `CudaEngine`. The merged state
(`state()`, own ranges concatenated; G and step from shard 0, identical everywhere) must hash equal to the
unsharded engine's; tests/test_cuda_sharding.py and experiments/s5_shard_equivalence.py --gpu check it
after every batch. Exchange volumes are counted in entries and bytes like the oracle does; exchange 1
entries are the NON-ZERO partial score entries (the oracle counts touched classes, which also includes
classes whose partial sum returned to 0, so the oracle's count is an upper bound of this one).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..api import DIAG_COUNTERS, STATE_FIELDS, CorpusArrays, state_hash_of, state_shapes
from ..config import EngineConfig
from .cuda_engine import U32_MAX, CudaEngine


class ShardedCudaEngine:
    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays, n_shards: int, **engine_kw: Any) -> None:
        if cfg.n_clauses % n_shards:
            raise ValueError("n_clauses must be a multiple of n_shards")
        if cfg.dense_equivalent or cfg.mode == "tabular":
            raise ValueError("sharding is defined for the sparse engine on a corpus")
        self.cfg, self.n_shards = cfg, int(n_shards)
        self.per = cfg.n_clauses // n_shards
        self.shards = [CudaEngine(cfg, corpus, own_range=(r * self.per, (r + 1) * self.per), **engine_kw) for r in range(n_shards)]
        self.cp = self.shards[0].cp
        self.corpus = corpus
        self.n_examples = self.shards[0].n_examples
        self.min_free = cfg.n_clauses // 128
        self.step = 0
        self.exchange = {"ex1_entries": 0, "ex1_bytes": 0, "ex2_entries": 0, "ex2_bytes": 0,
                         "ex3_entries": 0, "ex3_bytes": 0, "batches": 0,
                         "fired_classes": 0}   # sum over examples of |fired classes| (non-zero SUMMED scores), the term of the section 9 bound

    def owner(self, j: int) -> int:
        return int(j) // self.per

    def n_batches(self) -> int:
        return self.shards[0].n_batches()

    def n_free(self) -> int:
        return sum(sh.n_free() for sh in self.shards)

    # ------------------------------------------------------------------ one batch
    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        cp, sh0 = self.cp, self.shards[0]
        n_total = self.n_examples
        lo = int(batch_index) * self.cfg.batch
        if lo >= n_total or batch_index < 0:
            raise ValueError("batch_index out of range")
        hi = min(n_total, lo + self.cfg.batch)
        ids = cp.empty(hi - lo, dtype=cp.uint32)
        sh0._launch("k1_perm_ids", hi - lo, sh0._P, sh0._X, sh0._u32(epoch), np.uint64(lo), sh0._u32(hi - lo), np.uint64(n_total), ids)
        return self._train(epoch, ids)

    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        ids = np.ascontiguousarray(np.asarray(examples).reshape(-1), dtype=np.uint32)
        return self._train(epoch, self.cp.asarray(ids))

    def _train(self, epoch: int, ids_dev) -> dict[str, int]:
        cp, S, cfg = self.cp, self.shards, self.cfg
        n = int(ids_dev.shape[0])
        # spec 4 on every shard's own clauses, then EXCHANGE 1: summed scores and covers
        bts = [sh._phase_forward(epoch, ids_dev) for sh in S]
        scores = bts[0].fw.scores.copy()
        cover = bts[0].cover.copy()
        self.exchange["ex1_entries"] += int(cp.count_nonzero(bts[0].fw.scores).get())
        for bt in bts[1:]:
            self.exchange["ex1_entries"] += int(cp.count_nonzero(bt.fw.scores).get())
            scores += bt.fw.scores
            cover += bt.cover
        self.exchange["ex1_entries"] += self.n_shards * n  # the cover counts
        self.exchange["fired_classes"] += int(cp.count_nonzero(scores).get())
        for sh, bt in zip(S, bts):
            sh._phase_select(bt, scores, cover)
        # spec 5.1 to 5.5 locally
        for sh, bt in zip(S, bts):
            sh._phase_feedback(bt)
            sh._phase_gate_mass(bt)
        # spec 5.6 from the GLOBAL free list, lowest id first: EXCHANGE 2 (free counts)
        n_req = S[0]._phase_seed_literals(bts[0])
        for bt in bts[1:]:
            bt.req_i, bt.lits, bt.n_lits = bts[0].req_i, bts[0].lits, bts[0].n_lits
        free_r = [sh.n_free() for sh in S]
        self.exchange["ex2_entries"] += self.n_shards
        n_requests = n_seeded = 0
        assigned: list[list[int]] = [[] for _ in S]
        if n_req:
            ids_host = ids_dev.get()
            req_i = bts[0].req_i.get()
            n_lits = bts[0].n_lits.get()
            ys = self.corpus.tok[ids_host[req_i]]
            served: dict[int, int] = {}
            for t in range(n_req):
                y = int(ys[t])
                n_requests += 1
                if not any(free_r) or served.get(y, 0) >= cfg.seed_cap:
                    continue
                served[y] = served.get(y, 0) + 1                      # N4
                if int(n_lits[t]) == 0:
                    continue
                r = next(r for r in range(self.n_shards) if free_r[r] > 0)  # the lowest free id lies in the first shard with a free clause
                free_r[r] -= 1
                assigned[r].append(t)
                n_seeded += 1
        for sh, bt, which in zip(S, bts, assigned):
            sh._seed_serve_assigned(bt, which)
        for sh in S:
            sh._phase_refresh()
        # spec 5.7 with EXCHANGE 3
        if S[0].sweep_due():
            for sh, bt in zip(S, bts):
                sh._sweep_local(bt.diag)
            free_total = sum(sh.n_free_now() for sh in S)
            if free_total < self.min_free:
                lists = [sh._sweep_candidates(self.min_free) for sh in S]
                self.exchange["ex3_entries"] += sum(int(x.size) for x in lists)
                merged = np.sort(np.concatenate(lists))[: self.min_free - free_total]
                ids = (merged & U32_MAX).astype(np.uint32)
                for r, sh in enumerate(S):
                    mine = ids[(ids // self.per) == r]
                    sh._recycle_list(mine, bts[r].diag)
            for sh in S:
                sh._sweep_shift()
        diag = dict.fromkeys(DIAG_COUNTERS, 0)
        for sh, bt in zip(S, bts):
            for k, v in sh._finish_batch(bt).items():
                diag[k] += v
        diag["n_examples"] -= (self.n_shards - 1) * n   # counted once, like the oracle's shard 0
        diag["n_seed_requests"] += n_requests
        diag["n_seeded"] += n_seeded
        self.step += 1
        self.exchange["batches"] += 1
        self.exchange["ex1_bytes"] = self.exchange["ex1_entries"] * 8   # (class u16, score i32) padded to 8
        self.exchange["ex2_bytes"] = self.exchange["ex2_entries"] * 4
        self.exchange["ex3_bytes"] = self.exchange["ex3_entries"] * 8   # (stat_tp u32, id u32)
        return diag

    # ------------------------------------------------------------------ inference: partial scores summed (exchange 1 only)
    def predict_scores(self, epoch: int, examples: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
        out = self.shards[0].predict_scores(epoch, examples, hidden)
        for sh in self.shards[1:]:
            out += sh.predict_scores(epoch, examples, hidden)
        return out

    def fired_clauses(self, epoch: int, example: int, hidden: np.ndarray | None = None) -> np.ndarray:
        return np.sort(np.concatenate([sh.fired_clauses(epoch, example, hidden) for sh in self.shards]))

    # ------------------------------------------------------------------ merged state
    def state(self) -> dict[str, np.ndarray]:
        shapes = state_shapes(self.cfg)
        parts = [sh.state() for sh in self.shards]
        out = {}
        for name, dtype, kind in STATE_FIELDS:
            if kind == "1" or name == "G":
                out[name] = parts[0][name]
            else:
                out[name] = np.concatenate([p[name][r * self.per:(r + 1) * self.per] for r, p in enumerate(parts)]).reshape(shapes[name])
        out["step"] = np.array([self.step], dtype=np.uint64).reshape(shapes["step"])
        return out

    def state_hash(self) -> str:
        return state_hash_of(self.state(), self.cfg)

    def shards_agree_on_shared_state(self) -> bool:
        """G and step must be identical on every shard (computed, never exchanged)."""
        cp = self.cp
        g0 = self.shards[0].d["G"]
        return all(bool(cp.array_equal(sh.d["G"], g0)) and sh.step == self.shards[0].step for sh in self.shards)

    def exchange_bound_entries(self, fired_classes_per_example: float) -> float:
        """The per-batch bound of docs/11 section 9 in score entries: B * (K_NEG + 1 + |fired classes|)
        plus S integers plus S * MIN_FREE pairs [derived]. `fired_classes_per_example` is the mean number of
        classes with a non-zero summed score per example (`exchange["fired_classes"] / examples`); what one
        shard sends in exchange 1 (its non-zero partial entries) is at most B * |fired classes| of that."""
        return self.cfg.batch * (self.cfg.k_neg + 1 + fired_classes_per_example) + self.n_shards + self.n_shards * self.min_free
