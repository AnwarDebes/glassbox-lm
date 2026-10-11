"""Clause sharding on the oracle (docs/11 section 9; experiment s5, docs/05): a model of C clauses
trained as S shards, each an `OracleEngine` holding the full clause id space but OWNING only its
contiguous range [r*C/S, (r+1)*C/S). Foreign clauses are never alive, never free, never seeded on a
shard. Between the phases of spec section 5 the wrapper performs exactly the three exchanges of
section 9 and nothing else:

  exchange 1  summed scores S_i[c] and summed cover_i (after the partial forward pass)
  exchange 2  the free-clause count of every shard (seeding from the global free list)
  exchange 3  each shard's top-up candidate list, truncated to MIN_FREE (the sweep's global top-up)

Everything else runs unchanged on the shards through the phase methods of `OracleEngine`
(`_phase_forward`, `_forward_finish`, `_phase_feedback`, `_gate_mass`, `_seed_request`,
`_seed_literals`, `_seed_serve`, `_phase_refresh`, `_sweep_local`, `_sweep_candidates`,
`_sweep_shift`). The merged state must hash equal to the unsharded engine's state; s5 checks it.
Exchange volumes are counted in entries and bytes for the bound of section 9.
"""

from __future__ import annotations

import heapq

import numpy as np

from tmlm.engine.api import DIAG_COUNTERS, STATE_FIELDS, CorpusArrays, EngineConfig, state_hash_of, state_shapes
from tmlm.engine.config import Stream
from tmlm.engine.rng import perm
from tmlm.engine.oracle.oracle import OracleEngine


class ShardedOracle:
    def __init__(self, cfg: EngineConfig, corpus: CorpusArrays, n_shards: int) -> None:
        if cfg.n_clauses % n_shards:
            raise ValueError("n_clauses must be a multiple of n_shards")
        if cfg.dense_equivalent or cfg.mode == "tabular":
            raise ValueError("sharding is defined for the sparse engine on a corpus")
        self.cfg, self.n_shards = cfg, n_shards
        self.per = cfg.n_clauses // n_shards
        self.shards = [OracleEngine(cfg, corpus=corpus) for _ in range(n_shards)]
        for r, sh in enumerate(self.shards):
            lo, hi = r * self.per, (r + 1) * self.per
            sh.free = [j for j in sh.free if lo <= j < hi]  # ascending, a valid heap
            heapq.heapify(sh.free)
        self.n_examples = self.shards[0].n_examples
        self.min_free = self.shards[0].min_free
        self.step = 0
        self.exchange = {"ex1_entries": 0, "ex1_bytes": 0, "ex2_entries": 0, "ex2_bytes": 0,
                         "ex3_entries": 0, "ex3_bytes": 0, "batches": 0, "examples": 0,
                         "fired_classes": 0}  # fired_classes: distinct classes with a partial score, summed over examples (the bound's term)

    def owner(self, j: int) -> int:
        return j // self.per

    def n_batches(self) -> int:
        return self.shards[0].n_batches()

    # ------------------------------------------------------------------ one batch
    def train_batch(self, epoch: int, batch_index: int) -> dict[str, int]:
        sh0 = self.shards[0]
        n, b = self.n_examples, self.cfg.batch
        lo = batch_index * b
        if not 0 <= lo < n:
            raise ValueError("batch_index out of range")
        order = [perm(self.cfg.seed, int(Stream.PERM), epoch, x, n) for x in range(lo, min(n, lo + b))]
        ids = [int(sh0.corpus.targets[idx]) for idx in order]
        return self._train(epoch, ids)

    def train_explicit(self, epoch: int, examples: np.ndarray) -> dict[str, int]:
        return self._train(epoch, [int(v) for v in np.asarray(examples).reshape(-1)])

    def _train(self, epoch: int, ids: list[int]) -> dict[str, int]:
        S = self.shards
        for sh in S:
            sh.diag = dict.fromkeys(DIAG_COUNTERS, 0)
        S[0]._count("n_examples", len(ids))
        exs = [[sh._make_example(epoch, ident, None) for ident in ids] for sh in S]
        # spec 4 on every shard's own clauses, then EXCHANGE 1
        events = []
        for sh, ex_r in zip(S, exs):
            ev: dict[int, list[int]] = {}
            for i, ex in enumerate(ex_r):
                sh._forward_partial(ex)
                sh._count("n_candidates", ex.n_candidates)
                sh._count("n_fired", len(ex.fired))
                for j in ex.fired:
                    ev.setdefault(j, []).append(i)
            events.append(ev)
        for i in range(len(ids)):
            total: dict[int, int] = {}
            cover = 0
            for ex_r in exs:
                for c, v in ex_r[i].scores.items():
                    total[c] = total.get(c, 0) + v
                    self.exchange["ex1_entries"] += 1
                cover += ex_r[i].cover
            self.exchange["ex1_entries"] += self.n_shards  # the cover counts
            self.exchange["fired_classes"] += len(total)
            self.exchange["examples"] += 1
            for sh, ex_r in zip(S, exs):
                ex_r[i].scores = dict(total)
                ex_r[i].cover = cover
                sh._forward_finish(ex_r[i])
        # spec 5.1 to 5.5 locally
        changed = [sh._phase_feedback(ev, ex_r) for sh, ev, ex_r in zip(S, events, exs)]
        for sh, ex_r in zip(S, exs):
            sh._gate_mass(ex_r)
        # spec 5.6 from the GLOBAL free list, lowest id first: EXCHANGE 2 (free counts; here the heap tops)
        self.exchange["ex2_entries"] += self.n_shards
        served: dict[int, int] = {}
        for i in range(len(ids)):
            ex0 = exs[0][i]
            if not S[0]._seed_request(ex0):
                continue
            tops = [(sh.free[0], r) for r, sh in enumerate(S) if sh.free]
            if not tops or served.get(ex0.y, 0) >= self.cfg.seed_cap:
                continue
            served[ex0.y] = served.get(ex0.y, 0) + 1
            j, r = min(tops)
            lits = S[r]._seed_literals(exs[r][i])
            if not lits:
                continue
            heapq.heappop(S[r].free)
            S[r]._seed_serve(exs[r][i], j, lits)
            changed[r].add(j)
        for sh, ch in zip(S, changed):
            sh._phase_refresh(ch)
        # spec 5.7 with EXCHANGE 3
        if S[0].sweep_due():
            for sh in S:
                sh._sweep_local()
            free_total = sum(len(sh.free) for sh in S)
            if free_total < self.min_free:
                lists = [sh._sweep_candidates()[: self.min_free] for sh in S]
                self.exchange["ex3_entries"] += sum(len(x) for x in lists)
                merged = sorted(pair for x in lists for pair in x)[: self.min_free - free_total]
                for _, j in merged:
                    S[self.owner(j)]._recycle(j)
            for sh in S:
                sh._sweep_shift()
        for sh in S:
            sh.step += 1
        self.step += 1
        self.exchange["batches"] += 1
        self.exchange["ex1_bytes"] = self.exchange["ex1_entries"] * 8   # (class u16, score i32) padded to 8
        self.exchange["ex2_bytes"] = self.exchange["ex2_entries"] * 4
        self.exchange["ex3_bytes"] = self.exchange["ex3_entries"] * 8   # (stat_tp u32, id u32)
        diag = dict.fromkeys(DIAG_COUNTERS, 0)
        for sh in S:
            for k, v in sh.diag.items():
                diag[k] += v
        return diag

    # ------------------------------------------------------------------ merged state
    def state(self) -> dict[str, np.ndarray]:
        shapes = state_shapes(self.cfg)
        parts = [sh.state() for sh in self.shards]
        out = {}
        for name, dtype, kind in STATE_FIELDS:
            if kind == "1" or name == "G":
                out[name] = parts[0][name]  # step and G are identical on every shard
            else:
                out[name] = np.concatenate([p[name][r * self.per:(r + 1) * self.per] for r, p in enumerate(parts)]).reshape(shapes[name])
        out["step"] = np.array([self.step], dtype=np.uint64).reshape(shapes["step"])
        return out

    def state_hash(self) -> str:
        return state_hash_of(self.state(), self.cfg)

    def shards_agree_on_shared_state(self) -> bool:
        """G and step must be identical on every shard (they are computed, never exchanged)."""
        g0 = self.shards[0].G
        return all(sh.G == g0 and sh.step == self.shards[0].step for sh in self.shards)

    def exchange_bound_entries(self, fired_classes_per_example: float) -> float:
        """The per-batch bound of docs/11 section 9 in score entries: B * (K_NEG + 1 + |fired classes|)
        plus S integers plus S * MIN_FREE pairs [derived]."""
        return self.cfg.batch * (self.cfg.k_neg + 1 + fired_classes_per_example) + self.n_shards + self.n_shards * self.min_free
