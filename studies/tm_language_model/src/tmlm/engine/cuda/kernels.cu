/* kernels.cu: the CUDA engine of docs/11-engine-spec.md (docs/03 kernels K1..K14; docs/04 M4).
 *
 * All learning logic is in ../core/tm_core.h, compiled here by nvcc as __host__ __device__ code and
 * called from one-thread-per-item kernels. This file holds only the batch orchestration of docs/11
 * sections 4 and 5; the parts that are sorts, scans and compactions (K4, K7, K8, the index rebuild
 * of K12, the free list, the sweep's top-up list) run as CuPy array operations in cuda_engine.py.
 *
 * Determinism (docs/11 section 3, protocol rule 4): no curand, no floating point, no thread-id
 * dependence. Every kernel is a grid-stride loop over items whose result is a pure function of the
 * item; the only cross-thread writes are integer atomicAdd on class scores (K5), on the gate mass
 * G (K10) and on the diagnostic counters, all of which are commutative integer sums and therefore
 * order-independent. Events of a clause are applied sequentially in ascending example index by
 * the one thread that owns the clause (K9). Seeding (K11b) and nothing else is a single thread.
 * The result cannot depend on the grid or block size.
 *
 * Kernel map (docs/03): K1 k1_build_examples / k1_perm_ids; K2 k2_count_candidates; K3 k3_expand,
 * k3_verify (K4 compaction: CuPy nonzero); K5 k5_scores; K6 k6_select; K7, K8: CuPy sort and
 * segment bounds; K9 k9_feedback (lazy_owned) and k9_feedback_eager (N10), k9_dense_feedback
 * (spec 6); K10 k10_gate_mass; K11 k11a_seed_literals, k11b_seed_serve; K12 k12_refresh (+ CuPy
 * index sort); K13 k13_sweep, k13_recycle_list, k13_halve_stats; K14 predict = K1, K2, K3, K5 with
 * training = 0, plus k_fired_bruteforce for receipts and tests.
 */
#include "../core/tm_core.h"

#define TM_GRID_STRIDE(i, n) \
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += gridDim.x * blockDim.x)

typedef unsigned long long u64_t;

__device__ static inline void tm_flush_diag(const uint64_t *d, u64_t *diag) {
    for (int q = 0; q < TM_D_COUNT; ++q) if (d[q]) atomicAdd(&diag[q], (u64_t)d[q]);
}

extern "C" {

/* layout check: the host builds tm_params, tm_corpus, tm_state with ctypes and must match nvcc */
__global__ void k_sizes(u64_t *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = sizeof(tm_params); out[1] = sizeof(tm_corpus); out[2] = sizeof(tm_state); out[3] = sizeof(tm_example);
    }
}

/* ---- K1: examples of a batch ---- */
__global__ void k1_perm_ids(const tm_params *P, const tm_corpus *X, uint32_t epoch, u64_t lo, uint32_t n, u64_t n_total, uint32_t *ids) {
    TM_GRID_STRIDE(i, n) {
        uint64_t idx = tm_perm(P->seed, epoch, (uint64_t)lo + i, (uint64_t)n_total);
        ids[i] = P->mode == TM_MODE_TABULAR ? (uint32_t)idx : X->targets[idx];
    }
}
__global__ void k1_build_examples(const tm_params *P, const tm_corpus *X, tm_example *ex, uint32_t epoch,
                                  const uint32_t *ids, uint32_t n, const uint8_t *bitmap) {
    TM_GRID_STRIDE(i, n) tm_make_example(P, X, &ex[i], epoch, ids[i], bitmap);
}

/* ---- K2: posting-list lookups, one thread per (example, lookup slot) (oracle _candidates) ----
 * Language: slot s = 4 * (index into exact_seg_ids) + kind; a MASK or EDGE context looks up kind 0 only.
 * Tabular: slot s = feature f, looked up iff x[f] == 1. The index is (idx_keys ascending unique
 * anchor literals, idx_start CSR offsets into idx_clause, clause ids ascending inside a key). */
__global__ void k2_count_candidates(const tm_params *P, const tm_example *ex, uint32_t n, uint32_t n_slots,
                                    const uint32_t *idx_keys, const uint32_t *idx_start, uint32_t n_keys,
                                    uint32_t *slot_count, uint32_t *slot_first) {
    TM_GRID_STRIDE(t, n * n_slots) {
        uint32_t i = t / n_slots, s = t % n_slots;
        const tm_example *e = &ex[i];
        uint32_t lit = TM_LIT_EMPTY;
        if (P->mode == TM_MODE_TABULAR) {
            if (s < P->n_features && e->x[s] == 1) lit = tm_lit_pack(0, s, 0);
        } else {
            uint32_t si = s >> 2, k = s & 3;
            if (si < P->n_exact) {
                uint32_t seg = P->exact_seg_ids[si];
                const uint16_t *row = e->ctx[seg];
                int special = row[0] == P->sym_mask || row[0] == P->sym_edge;
                if (k == 0 || !special) lit = tm_lit_pack(seg, row[k], 0);
            }
        }
        uint32_t cnt = 0, first = 0;
        if (lit != TM_LIT_EMPTY && n_keys) {
            uint32_t pos = tm_lower_bound(idx_keys, 0, n_keys, lit);
            if (pos < n_keys && idx_keys[pos] == lit) { first = idx_start[pos]; cnt = idx_start[pos + 1] - first; }
        }
        slot_count[t] = cnt; slot_first[t] = first;
    }
}

/* ---- K3: expand the lookups into (example, clause) candidates, then verify each ---- */
__global__ void k3_expand(uint32_t n_total_slots, uint32_t n_slots, const uint32_t *slot_count, const uint32_t *slot_first,
                          const uint32_t *slot_off, const uint32_t *idx_clause, uint32_t *cand_ex, uint32_t *cand_j) {
    TM_GRID_STRIDE(t, n_total_slots) {
        uint32_t c = slot_count[t];
        if (!c) continue;
        uint32_t o = slot_off[t], f = slot_first[t], i = t / n_slots;
        for (uint32_t k = 0; k < c; ++k) { cand_ex[o + k] = i; cand_j[o + k] = idx_clause[f + k]; }
    }
}
__global__ void k3_verify(const tm_params *P, const tm_corpus *X, const tm_state *S, const tm_example *ex,
                          uint32_t n_cand, const uint32_t *cand_ex, const uint32_t *cand_j, uint8_t *fired, u64_t *n_inc_sum) {
    uint64_t inc = 0;
    TM_GRID_STRIDE(t, n_cand) {
        uint32_t j = cand_j[t];
        fired[t] = tm_fires(P, X, S, &ex[cand_ex[t]], j) ? 1 : 0;
        inc += S->n_inc[j];
    }
    if (n_inc_sum && inc) atomicAdd(n_inc_sum, (u64_t)inc);
}
/* dense-equivalent forward and receipts: one thread per (example, clause) pair, out[i * C + j]
 * (oracle _fired in dense mode, _fires_live for brute-force receipts) */
__global__ void k_fired_bruteforce(const tm_params *P, const tm_corpus *X, const tm_state *S, const tm_example *ex, uint32_t n,
                                   int dense, int training, uint8_t *out) {
    TM_GRID_STRIDE(t, n * P->C) {
        uint32_t i = t / P->C, j = t % P->C;
        out[t] = (dense ? tm_fires_dense(P, X, S, &ex[i], j, training) : tm_fires_live(P, X, S, &ex[i], j)) ? 1 : 0;
    }
}

/* ---- K5: class scores, integer atomics (commutative, order-free) (oracle _scores) ---- */
__global__ void k5_scores(const tm_params *P, const tm_state *S, uint32_t n_fired, const uint32_t *fired_ex, const uint32_t *fired_j, int32_t *scores) {
    TM_GRID_STRIDE(t, n_fired) {
        uint32_t j = fired_j[t];
        const uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS; const int16_t *w = S->vote_w + (uint64_t)j * P->V_SLOTS;
        int32_t *row = scores + (uint64_t)fired_ex[t] * P->V;
        for (uint32_t k = 0; k < P->V_SLOTS; ++k) if (cls[k] != TM_CLS_EMPTY) atomicAdd(&row[cls[k]], (int32_t)w[k]);
    }
}

/* ---- K6: per example. k6_cover: the (partial) cover count over this engine's fired clauses; k6_select: S_y,
 * target gate mass, focused negatives and the seed request from the SUMMED scores and cover (spec 4 step 5,
 * 5.6; docs/11 section 9 exchange 1: in a sharded run the scores and covers of all shards are summed between
 * the two kernels, so k6_select is a pure function of the summed inputs and identical on every shard) ---- */
__global__ void k6_cover(const tm_params *P, const tm_state *S, const tm_example *ex, uint32_t n, const uint32_t *fired_off,
                         const uint32_t *fired_j, uint32_t *cover) {
    TM_GRID_STRIDE(i, n) {
        uint32_t c = 0;
        for (uint32_t t = fired_off[i]; t < fired_off[i + 1]; ++t) if (tm_owns(P, S, fired_j[t], ex[i].y)) c += 1;
        cover[i] = c;
    }
}
__global__ void k6_select(const tm_params *P, tm_example *ex, uint32_t n, const uint32_t *cover, const int32_t *scores, uint8_t *seed_req) {
    TM_GRID_STRIDE(i, n) {
        tm_example *e = &ex[i];
        const int32_t *row = scores + (uint64_t)i * P->V;
        e->s_y = row[e->y];
        e->py_q32 = tm_py_q32(P, e->s_y);
        e->cover = cover[i];
        e->n_neg = 0;
        if (P->neg_mode == TM_NEG_FOCUSED && !P->dense_equivalent) {
            /* up to K_NEG classes c != y with the largest S > 0, ties toward the lower class id;
             * classes are visited in ascending order, so a tie never displaces an earlier class */
            int32_t bs[TM_MAX_K_NEG]; uint32_t bc[TM_MAX_K_NEG]; uint32_t nb = 0;
            for (uint32_t c = 0; c < P->V; ++c) {
                int32_t s = row[c];
                if (c == e->y || s <= 0) continue;
                uint32_t pos = nb;
                while (pos > 0 && bs[pos - 1] < s) --pos;
                if (pos >= P->K_NEG) continue;
                uint32_t last = nb < P->K_NEG ? nb : P->K_NEG - 1;
                for (uint32_t q = last; q > pos; --q) { bs[q] = bs[q - 1]; bc[q] = bc[q - 1]; }
                bs[pos] = s; bc[pos] = c;
                if (nb < P->K_NEG) ++nb;
            }
            for (uint32_t q = 0; q < nb; ++q) { e->neg_cls[q] = (uint16_t)bc[q]; e->neg_s[q] = bs[q]; e->neg_pc_q32[q] = tm_pc_q32(P, bs[q]); }
            e->n_neg = nb;
        }
        uint32_t p_req = (uint32_t)(((uint64_t)e->py_q32 * (uint64_t)P->rho_seed_q32) >> 32);
        seed_req[i] = (e->cover < P->K_COVER && tm_bern(p_req, P->seed, TM_S_SEED, e->a, 0, 0)) ? 1 : 0;
    }
}

/* ---- K9: feedback, one thread per touched clause, events in ascending i (spec 5.1 to 5.4) ---- */
__global__ void k9_feedback(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t n_touched,
                            const uint32_t *touched_j, const uint32_t *touched_start, const uint32_t *touched_end,
                            const uint32_t *ev_i, const int32_t *scores, uint8_t *changed, u64_t *diag) {
    uint64_t d[TM_D_COUNT]; for (int q = 0; q < TM_D_COUNT; ++q) d[q] = 0;
    TM_GRID_STRIDE(t, n_touched) {
        uint32_t j = touched_j[t], s = touched_start[t], cnt = touched_end[t] - s;
        tm_settle(P, S, j);
        uint32_t cap = cnt < P->E_CAP ? cnt : P->E_CAP;
        for (uint32_t k = 0; k < cap; ++k) { uint32_t i = ev_i[s + k]; tm_event(P, X, S, &ex[i], j, scores + (uint64_t)i * P->V, d); }
        d[TM_D_N_EVENTS_DROPPED_CAP] += cnt > P->E_CAP ? cnt - P->E_CAP : 0;
        changed[j] = 1;
    }
    tm_flush_diag(d, diag);
}
/* N10: eager_exact in the sparse engine, one thread per clause alive at batch start */
__global__ void k9_feedback_eager(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t n,
                                  const uint32_t *ev_ptr, const uint32_t *ev_cnt, const uint32_t *ev_i, const int32_t *scores,
                                  uint8_t *changed, u64_t *diag) {
    uint64_t d[TM_D_COUNT]; for (int q = 0; q < TM_D_COUNT; ++q) d[q] = 0;
    TM_GRID_STRIDE(j, P->C) {
        if (!S->alive[j]) continue;
        uint32_t ptr = ev_ptr[j], cnt = ev_cnt[j], place = 0;
        for (uint32_t i = 0; i < n; ++i) {
            int in_list = ptr != TM_U32_MAX && place < cnt && ev_i[ptr + place] == i;
            if (!in_list) {
                if (tm_owns(P, S, j, ex[i].y) && tm_bern(ex[i].py_q32, P->seed, TM_S_GATE_TGT, ex[i].a, j, 0)) {
                    d[TM_D_N_TGT_GATE_PASS] += 1; tm_type_ib(P, S, &ex[i], j, d);
                }
            } else {
                if (place < P->E_CAP) tm_event(P, X, S, &ex[i], j, scores + (uint64_t)i * P->V, d);
                else d[TM_D_N_EVENTS_DROPPED_CAP] += 1;
                ++place;
            }
        }
        changed[j] = 1;
    }
    tm_flush_diag(d, diag);
}
/* spec 6: dense-equivalent feedback, one thread per clause, one example */
__global__ void k9_dense_feedback(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex,
                                  const uint8_t *fires, const int32_t *scores, u64_t *diag) {
    uint64_t d[TM_D_COUNT]; for (int q = 0; q < TM_D_COUNT; ++q) d[q] = 0;
    TM_GRID_STRIDE(j, P->C) tm_dense_clause_feedback(P, X, S, ex, j, fires[j], scores, d);
    tm_flush_diag(d, diag);
}

/* ---- K10: gate mass (spec 5.5), commutative u64 sums ---- */
__global__ void k10_gate_mass(const tm_state *S, const tm_example *ex, uint32_t n) {
    TM_GRID_STRIDE(i, n) atomicAdd((u64_t *)&S->G[ex[i].y], (u64_t)(ex[i].py_q32 >> 16));
}

/* ---- K11: seeding (spec 5.6). Literal draws depend on the example only (parallel); serving is
 * sequential in ascending i with the per-class cap and the lowest free id first (one thread). ---- */
__global__ void k11a_seed_literals(const tm_params *P, const tm_corpus *X, const tm_example *ex, uint32_t n_req,
                                   const uint32_t *req_i, uint32_t *lits, uint32_t *n_lits) {
    TM_GRID_STRIDE(t, n_req) n_lits[t] = tm_seed_literals(P, X, &ex[req_i[t]], lits + 3 * t);
}
__global__ void k11b_seed_serve(const tm_params *P, tm_state *S, const tm_example *ex, uint32_t n_req, const uint32_t *req_i,
                                const uint32_t *lits, const uint32_t *n_lits, uint32_t *served, const uint32_t *free_ids,
                                uint32_t free_n, uint32_t *free_head, uint8_t *changed, u64_t *diag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t head = *free_head;
    uint64_t n_requests = 0, n_seeded = 0;
    for (uint32_t t = 0; t < n_req; ++t) {
        const tm_example *e = &ex[req_i[t]];
        n_requests += 1;
        if (head >= free_n || served[e->y] >= P->SEED_CAP) continue;
        served[e->y] += 1;                                     /* N4 */
        if (n_lits[t] == 0) continue;
        uint32_t j = free_ids[head++];
        tm_seed_clause(P, S, e, j, lits + 3 * t, n_lits[t]);
        changed[j] = 1;
        n_seeded += 1;
    }
    *free_head = head;
    if (n_requests) atomicAdd(&diag[TM_D_N_SEED_REQUESTS], (u64_t)n_requests);
    if (n_seeded) atomicAdd(&diag[TM_D_N_SEEDED], (u64_t)n_seeded);
}

/* sharded seeding (docs/11 section 9, exchange 2): the host has already applied the per-class cap and the
 * global lowest-free-id rule and hands this shard the requests it serves, in ascending i, each with at least
 * one literal; the shard pops its own free list in order. No cap check and no request counting here. */
__global__ void k11c_seed_serve_assigned(const tm_params *P, tm_state *S, const tm_example *ex, uint32_t n_req, const uint32_t *req_i,
                                         const uint32_t *lits, const uint32_t *n_lits, const uint32_t *free_ids, uint32_t free_n,
                                         uint32_t *free_head, uint8_t *changed) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t head = *free_head;
    for (uint32_t t = 0; t < n_req && head < free_n; ++t) {
        uint32_t j = free_ids[head++];
        tm_seed_clause(P, S, &ex[req_i[t]], j, lits + 3 * t, n_lits[t]);
        changed[j] = 1;
    }
    *free_head = head;
}

/* ---- K12: refresh of changed clauses (spec 2). The index is then rebuilt from `anchor` by CuPy. ---- */
__global__ void k12_refresh(const tm_params *P, const tm_corpus *X, tm_state *S, uint8_t *changed, int all, uint32_t *anchor_changed) {
    uint32_t any = 0;
    TM_GRID_STRIDE(j, P->C) {
        if (!all && !changed[j]) continue;
        uint32_t a = tm_refresh_clause(P, X, S, j);
        if (a != S->anchor[j]) { S->anchor[j] = a; any = 1; }
        changed[j] = 0;
    }
    if (any) atomicOr(anchor_changed, 1u);
}

/* ---- K13: sweep (spec 5.7), per-clause part; the MIN_FREE top-up list is built by CuPy ---- */
__global__ void k13_sweep(const tm_params *P, const tm_corpus *X, tm_state *S, u64_t *diag) {
    uint64_t d[TM_D_COUNT]; for (int q = 0; q < TM_D_COUNT; ++q) d[q] = 0;
    uint32_t period = P->SWEEP_PERIOD;
    TM_GRID_STRIDE(j, P->C) {
        if (!S->alive[j]) continue;
        tm_settle(P, S, j);
        uint32_t a = tm_refresh_clause(P, X, S, j);
        uint32_t age = (uint32_t)(S->step & TM_U32_MAX) - S->last_tp_step[j];       /* N1: 32-bit wrap */
        if (a == TM_ANCHOR_NONE || !tm_owns_any(P, S, j) || age > P->DEAD_SWEEPS * period) {
            tm_recycle(P, S, j);
            a = tm_refresh_clause(P, X, S, j);
            d[TM_D_N_RECYCLED] += 1;
        }
        S->anchor[j] = a;
    }
    tm_flush_diag(d, diag);
}
__global__ void k13_recycle_list(const tm_params *P, const tm_corpus *X, tm_state *S, uint32_t n, const uint32_t *ids, u64_t *diag) {
    uint64_t d[TM_D_COUNT]; for (int q = 0; q < TM_D_COUNT; ++q) d[q] = 0;
    TM_GRID_STRIDE(t, n) { uint32_t j = ids[t]; tm_recycle(P, S, j); S->anchor[j] = tm_refresh_clause(P, X, S, j); d[TM_D_N_RECYCLED] += 1; }
    tm_flush_diag(d, diag);
}
__global__ void k13_halve_stats(const tm_params *P, tm_state *S) {
    TM_GRID_STRIDE(j, P->C) { S->stat_fire[j] >>= 1; S->stat_tp[j] >>= 1; }
}

/* ---- spec 6: dense-equivalent initial state (engine.cpp init_dense) ---- */
__global__ void k_init_dense(const tm_params *P, tm_state *S) {
    TM_GRID_STRIDE(j, P->C) {
        for (uint32_t f = 0; f < P->n_features; ++f) {
            S->lit_id[(uint64_t)j * P->L_SLOTS + 2 * f] = tm_lit_pack(0, f, 0);
            S->lit_id[(uint64_t)j * P->L_SLOTS + 2 * f + 1] = tm_lit_pack(0, f, 1);
            S->lit_state[(uint64_t)j * P->L_SLOTS + 2 * f] = TM_STATE_DENSE_INIT;
            S->lit_state[(uint64_t)j * P->L_SLOTS + 2 * f + 1] = TM_STATE_DENSE_INIT;
        }
        for (uint32_t c = 0; c < P->V; ++c) {
            S->vote_cls[(uint64_t)j * P->V_SLOTS + c] = (uint16_t)c;
            S->vote_w[(uint64_t)j * P->V_SLOTS + c] = (tm_rng32(P->seed, TM_S_SEED, j, c, 0) & 1u) ? 1 : -1;
        }
        S->alive[j] = 1;
    }
}

} /* extern "C" */
