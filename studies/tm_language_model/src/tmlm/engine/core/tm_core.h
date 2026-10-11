/* tm_core.h: the ONE header shared by the C++ host engine (g++) and the CUDA engine (nvcc).
 *
 * Normative text: docs/11-engine-spec.md v0.2 including section 8a (the oracle's readings N1 to
 * N12). Reference implementation: src/tmlm/engine/oracle/oracle.py. Every routine here mirrors
 * one oracle method (named in its comment) and must reproduce it bit for bit: the three
 * engines are compared on state_hash() and on DIAG_COUNTERS.
 *
 * What lives here: constants, packing, randomness, literal truth, and every PER-CLAUSE routine
 * of spec section 5 (events, Type Ia/Ib/II, insertion, candidate sampling, negatives, settle,
 * vote_add, seeding of one clause, recycling, refresh of one clause). What does NOT live here:
 * batch orchestration (candidate lookup through the inverted index, grouping events by clause,
 * gate mass, seeding order, sweep order, index rebuild). The CPU engine does that with OpenMP,
 * the CUDA engine with kernels; both call these routines.
 *
 * Integer-only. No floating point anywhere in learning. All argument integers are fixed width.
 */
#ifndef TMLM_TM_CORE_H
#define TMLM_TM_CORE_H

#include <stdint.h>

#ifdef __CUDACC__
#define TM_FN __host__ __device__ static inline
#else
#define TM_FN static inline
#endif

/* ---- constants (docs/11 sections 1, 2, 7; config.py) ---- */
#define TM_LIT_EMPTY        0xFFFFFFFFu
#define TM_ANCHOR_NONE      0xFFFFFFFFu
#define TM_CLS_EMPTY        0xFFFFu
#define TM_N_EXACT_SEG      16
#define TM_MAX_BAG_K        13
#define TM_MAX_SEG          (TM_N_EXACT_SEG + 2 * TM_MAX_BAG_K)   /* 42 */
#define TM_CLUSTER_SIZE_1   16
#define TM_CLUSTER_SIZE_2   256
#define TM_CLUSTER_SIZE_3   4096
#define TM_INCLUDE_THRESHOLD 128
#define TM_STATE_INSERT     123
#define TM_STATE_SEED       136
#define TM_STATE_DENSE_INIT 127
#define TM_W_MAX            32767
#define TM_MAX_L_SLOTS      64
#define TM_MAX_V_SLOTS      1024   /* raised from 64 on 2026-09-23 for the wide-vote tier test (docs/03: [C_wide, 1024]); no array is sized by it */
#define TM_MAX_K_NEG        8
#define TM_Q32_MAX          0xFFFFFFFFu
#define TM_U32_MAX          0xFFFFFFFFu
#define TM_NEG_CLASS_DRAW_BASE 65536u   /* spec 5.3 uniform_binomial: c = 65536 + class */

/* modes (config.MODES order), negative modes, decay modes */
enum tm_mode { TM_MODE_MASKED = 0, TM_MODE_DIFFUSION = 1, TM_MODE_CAUSAL = 2, TM_MODE_TABULAR = 3 };
enum tm_neg_mode { TM_NEG_FOCUSED = 0, TM_NEG_UNIFORM_BINOMIAL = 1, TM_NEG_BINOMIAL = 2 };
#define TM_NEG_K_MAX 32
enum tm_decay_mode { TM_DECAY_LAZY_OWNED = 0, TM_DECAY_EAGER_EXACT = 1 };

/* ---- RNG streams (docs/11 section 3; config.Stream) ---- */
enum tm_stream {
    TM_S_MASK = 1, TM_S_RATE = 2, TM_S_GATE_TGT = 3, TM_S_GATE_NEG = 4, TM_S_TYPE1_DEC = 5,
    TM_S_INSERT = 6, TM_S_INSERT_PICK = 7, TM_S_NEG_INSERT = 8, TM_S_NEG_PICK = 9,
    TM_S_SEED = 10, TM_S_SEED_PICK = 11, TM_S_DECAY = 12, TM_S_VETO = 13, TM_S_PERM = 14,
    TM_S_SAMPLER = 15, TM_S_NEG_DRAW = 16
};

/* diagnostics (api.DIAG_COUNTERS order) */
enum tm_diag {
    TM_D_N_EXAMPLES = 0, TM_D_N_CANDIDATES, TM_D_N_FIRED, TM_D_N_EVENTS_APPLIED,
    TM_D_N_EVENTS_SKIPPED_NOT_FIRING, TM_D_N_EVENTS_DROPPED_CAP, TM_D_N_TGT_GATE_PASS,
    TM_D_N_NEG_GATE_PASS, TM_D_N_TYPE1, TM_D_N_TYPE2, TM_D_N_INSERTED, TM_D_N_NEG_INSERTED,
    TM_D_N_VETOES, TM_D_N_SEED_REQUESTS, TM_D_N_SEEDED, TM_D_N_RECYCLED, TM_D_COUNT
};

/* ---- literal packing (docs/11 section 1.4): lit = (seg << 17) | (sym << 1) | neg ---- */
TM_FN uint32_t tm_lit_pack(uint32_t seg, uint32_t sym, uint32_t neg) { return (seg << 17) | (sym << 1) | neg; }
TM_FN uint32_t tm_lit_seg(uint32_t lit) { return lit >> 17; }
TM_FN uint32_t tm_lit_sym(uint32_t lit) { return (lit >> 1) & 0xFFFFu; }
TM_FN uint32_t tm_lit_neg(uint32_t lit) { return lit & 1u; }
TM_FN int32_t tm_seg_offset(uint32_t seg) { return seg < 8 ? (int32_t)seg - 8 : (int32_t)seg - 7; }
TM_FN uint32_t tm_bag_bucket(uint32_t seg) { return (seg - TM_N_EXACT_SEG) / 2; }
TM_FN uint32_t tm_bag_side(uint32_t seg) { return (seg - TM_N_EXACT_SEG) % 2; }
TM_FN uint32_t tm_bag_lo(uint32_t k) { return 8u * (1u << k) + 1u; }
TM_FN uint32_t tm_bag_hi(uint32_t k) { return 8u * (1u << (k + 1)); }

/* ---- deterministic randomness (docs/11 section 3; rng.py is the reference) ---- */
TM_FN uint64_t tm_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
TM_FN uint64_t tm_rng64(uint64_t seed, uint64_t stream, uint64_t a, uint64_t b, uint64_t c) {
    return tm_mix64(tm_mix64(tm_mix64(tm_mix64(seed ^ (stream * 0xD6E8FEB86659FD93ull)) + a) + b) + c);
}
TM_FN uint32_t tm_rng32(uint64_t seed, uint64_t stream, uint64_t a, uint64_t b, uint64_t c) {
    return (uint32_t)(tm_rng64(seed, stream, a, b, c) >> 32);
}
TM_FN int tm_bern(uint32_t p_q32, uint64_t seed, uint64_t stream, uint64_t a, uint64_t b, uint64_t c) {
    return tm_rng32(seed, stream, a, b, c) < p_q32;
}
TM_FN uint32_t tm_pick(uint32_t n, uint64_t seed, uint64_t stream, uint64_t a, uint64_t b, uint64_t c) {
    return (uint32_t)(((uint64_t)tm_rng32(seed, stream, a, b, c) * (uint64_t)n) >> 32);
}
/* epoch permutation: 4-round Feistel with cycle walking (rng.perm) */
TM_FN uint32_t tm_perm_bits(uint64_t n) { uint32_t b = 2; while ((1ull << b) < n) b += 2; return b; }
TM_FN uint64_t tm_perm(uint64_t seed, uint64_t epoch, uint64_t x, uint64_t n) {
    uint32_t h = tm_perm_bits(n) / 2;
    uint64_t m = (1ull << h) - 1ull;
    for (;;) {
        uint64_t left = x >> h, right = x & m;
        for (uint32_t r = 0; r < 4; ++r) {
            uint64_t nl = right;
            uint64_t nr = left ^ ((uint64_t)tm_rng32(seed, TM_S_PERM, epoch, r, right) & m);
            left = nl; right = nr;
        }
        x = (left << h) | right;
        if (x < n) return x;
    }
}

/* ---- parameters (config.EngineConfig + Derived), filled on the host once ---- */
typedef struct tm_params {
    uint32_t C, V, L_SLOTS, V_SLOTS, BUDGET, K_NEG, K_COVER, SEED_CAP, E_CAP, SWEEP_PERIOD, DEAD_SWEEPS;
    int32_t  T, MARGIN;
    uint32_t inv_s_q32, p_ins_q32, p_neg_q32, p_veto_q32, rho_seed_q32, mask_rate_q32, q_neg_q32;
    uint32_t kind_t70, kind_t80, kind_t90, nlit_t1, nlit_t2;
    uint64_t seed;
    uint64_t s_num, s_den;          /* exact rational value of s (oracle N6) */
    uint32_t mode, neg_mode, decay_mode, dense_equivalent;
    uint32_t n_bag_k, n_features;   /* n_features: tabular only */
    uint32_t sym_mask, sym_edge, n_sym;
    uint32_t min_free;              /* C / 128 */
    int32_t  neg_budget;            /* -1: none; else max included negated literals per clause */
    uint32_t cand_tries;            /* N15: candidate draws per sampling, keep the rarest symbol */
    uint64_t enabled_segs;          /* bit s set iff segment s is enabled in this mode */
    /* candidate sampling tables (config.derive): cumulative weights */
    uint32_t n_segs, seg_total;     uint32_t seg_ids[TM_MAX_SEG], seg_cum[TM_MAX_SEG];
    uint32_t n_exact, exact_total;  uint32_t exact_seg_ids[TM_N_EXACT_SEG], exact_cum[TM_N_EXACT_SEG];
    uint32_t neg_cdf_q32[TM_NEG_K_MAX + 1]; /* binomial mode: CDF of Binomial(V-1, Q/(V-1)) */
} tm_params;

/* ---- corpus (api.CorpusArrays) plus occurrence lists for bag literals, or tabular rows ---- */
typedef struct tm_corpus {
    const uint16_t *tok; uint64_t n_tok;
    const uint32_t *doc_start; uint32_t n_doc;
    const uint16_t *kind_map;       /* [4][V] */
    const uint8_t  *stop;           /* [V] */
    const uint32_t *sym_freq;       /* [N_SYM] */
    const uint32_t *targets; uint32_t n_targets;
    /* occurrence lists: for symbol s (any kind), positions q with kind_map[kind(s)][tok[q]] == s and
     * stop[tok[q]] == 0, ascending. occ_ptr[N_SYM + 1], occ_pos[occ_ptr[N_SYM]]. Evaluating a bag
     * literal through them is identical to the oracle's scan of the bucket. */
    const uint32_t *occ_ptr; const uint32_t *occ_pos;
    /* tabular mode */
    const uint8_t *x; const uint16_t *y; uint32_t n_rows;
} tm_corpus;

/* ---- model state (docs/11 section 2): structure-of-arrays, identical layout on CPU and GPU ---- */
typedef struct tm_state {
    uint32_t *lit_id;        /* [C][L_SLOTS] */
    uint8_t  *lit_state;     /* [C][L_SLOTS] */
    uint16_t *vote_cls;      /* [C][V_SLOTS] */
    int16_t  *vote_w;        /* [C][V_SLOTS] */
    uint64_t *decay_mark;    /* [C] */
    uint64_t *own_fire_mass; /* [C] */
    uint32_t *stat_fire;     /* [C] */
    uint32_t *stat_tp;       /* [C] */
    uint32_t *birth_step;    /* [C] */
    uint32_t *last_tp_step;  /* [C] */
    uint8_t  *alive;         /* [C] */
    uint64_t *G;             /* [V] */
    uint64_t  step;
    /* derived, never hashed */
    uint32_t *inc_lit;       /* [C][L_SLOTS] included literals in the 4 groups, ascending inside each */
    uint8_t  *n_inc;         /* [C] */
    uint8_t  *n_inc_neg;     /* [C] included negated literals (neg_budget) */
    uint32_t *anchor;        /* [C] */
} tm_state;

/* ---- one example (oracle _Example) ---- */
typedef struct tm_example {
    uint64_t a; uint32_t e; uint32_t p; uint32_t y;
    uint32_t doc_lo, doc_hi;
    uint16_t ctx[TM_N_EXACT_SEG][4];
    const uint8_t *bitmap;       /* frozen hidden bitmap over all positions, or 0 */
    uint32_t r_a_q32;            /* diffusion rate */
    const uint8_t *x;            /* tabular row */
    /* forward-pass results (spec 4 step 5) */
    int32_t s_y; uint32_t py_q32; uint32_t cover;
    uint32_t n_neg; uint16_t neg_cls[TM_MAX_K_NEG]; uint32_t neg_pc_q32[TM_MAX_K_NEG]; int32_t neg_s[TM_MAX_K_NEG];
} tm_example;

/* ---- symbol kinds (config.kind_of) ---- */
TM_FN int tm_kind_of(const tm_params *P, uint32_t sym) {
    if (sym < P->V) return 0;
    uint32_t b = P->V;
    if (sym < b + TM_CLUSTER_SIZE_1) return 1;
    b += TM_CLUSTER_SIZE_1;
    if (sym < b + TM_CLUSTER_SIZE_2) return 2;
    b += TM_CLUSTER_SIZE_2;
    if (sym < b + TM_CLUSTER_SIZE_3) return 3;
    return -1;
}

/* ---- hiding and context (spec 1.3; oracle _hidden, _ctx) ---- */
TM_FN int tm_hidden(const tm_params *P, const tm_example *ex, uint32_t q) {
    if (ex->bitmap) return ex->bitmap[q] != 0;
    if (P->mode == TM_MODE_MASKED) return tm_rng32(P->seed, TM_S_MASK, ex->e, q, 0) < P->mask_rate_q32;
    if (P->mode == TM_MODE_DIFFUSION) return tm_rng32(P->seed, TM_S_MASK, ex->a, q, 0) < ex->r_a_q32;
    return 0;
}
TM_FN uint32_t tm_ctx(const tm_params *P, const tm_corpus *X, const tm_example *ex, int64_t q, uint32_t k) {
    if (q < (int64_t)ex->doc_lo || q >= (int64_t)ex->doc_hi) return P->sym_edge;
    if ((uint32_t)q == ex->p || tm_hidden(P, ex, (uint32_t)q)) return P->sym_mask;
    return X->kind_map[k * P->V + X->tok[q]];
}

/* Build an example from a position (or a tabular row); the forward results are left zero. */
TM_FN void tm_make_example(const tm_params *P, const tm_corpus *X, tm_example *ex, uint32_t epoch,
                           uint32_t ident, const uint8_t *bitmap) {
    ex->e = epoch; ex->p = ident; ex->a = ((uint64_t)epoch << 40) | (uint64_t)ident;
    ex->bitmap = 0; ex->r_a_q32 = 0; ex->x = 0; ex->doc_lo = ex->doc_hi = 0;
    ex->s_y = 0; ex->py_q32 = 0; ex->cover = 0; ex->n_neg = 0;
    if (P->mode == TM_MODE_TABULAR) {
        ex->x = X->x + (uint64_t)ident * P->n_features;
        ex->y = X->y[ident];
        return;
    }
    ex->y = X->tok[ident];
    /* document: largest d with doc_start[d] <= ident */
    uint32_t lo = 0, hi = X->n_doc;      /* doc_start has n_doc + 1 entries */
    while (hi - lo > 1) { uint32_t mid = (lo + hi) / 2; if (X->doc_start[mid] <= ident) lo = mid; else hi = mid; }
    ex->doc_lo = X->doc_start[lo]; ex->doc_hi = X->doc_start[lo + 1];
    if (P->mode == TM_MODE_MASKED || P->mode == TM_MODE_DIFFUSION) {
        ex->bitmap = bitmap;
        if (P->mode == TM_MODE_DIFFUSION && !bitmap) ex->r_a_q32 = tm_rng32(P->seed, TM_S_RATE, ex->a, 0, 0);
    }
    for (uint32_t seg = 0; seg < TM_N_EXACT_SEG; ++seg) {
        if (!((P->enabled_segs >> seg) & 1ull)) { ex->ctx[seg][0] = ex->ctx[seg][1] = ex->ctx[seg][2] = ex->ctx[seg][3] = 0; continue; }
        int64_t q = (int64_t)ident + tm_seg_offset(seg);
        for (uint32_t k = 0; k < 4; ++k) ex->ctx[seg][k] = (uint16_t)tm_ctx(P, X, ex, q, k);
    }
}

/* ---- literal truth (spec 1.4; oracle _lit_true, _exact_true, _bag_true) ---- */
TM_FN uint32_t tm_lower_bound(const uint32_t *arr, uint32_t lo, uint32_t hi, uint32_t v) {
    while (lo < hi) { uint32_t mid = lo + (hi - lo) / 2; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
}
TM_FN int tm_positive_true(const tm_params *P, const tm_corpus *X, const tm_example *ex, uint32_t seg, uint32_t sym) {
    if (!((P->enabled_segs >> seg) & 1ull)) return 0;                 /* N11 */
    if (P->mode == TM_MODE_TABULAR) return sym < P->n_features && ex->x[sym] == 1;
    if (seg < TM_N_EXACT_SEG) {
        if (sym == P->sym_mask || sym == P->sym_edge) return ex->ctx[seg][0] == sym;
        int kind = tm_kind_of(P, sym);
        return kind >= 0 && ex->ctx[seg][kind] == sym;
    }
    int kind = tm_kind_of(P, sym);
    if (kind < 0) return 0;
    uint32_t k = tm_bag_bucket(seg), side = tm_bag_side(seg);
    uint32_t lo = tm_bag_lo(k), hi = tm_bag_hi(k);
    int64_t first, last;
    if (side == 0) { first = (int64_t)ex->p - (int64_t)hi; last = (int64_t)ex->p - (int64_t)lo; }
    else           { first = (int64_t)ex->p + (int64_t)lo; last = (int64_t)ex->p + (int64_t)hi; }
    if (first < (int64_t)ex->doc_lo) first = ex->doc_lo;
    if (last > (int64_t)ex->doc_hi - 1) last = (int64_t)ex->doc_hi - 1;
    if (first > last) return 0;
    uint32_t b = X->occ_ptr[sym], e = X->occ_ptr[sym + 1];
    uint32_t i = tm_lower_bound(X->occ_pos, b, e, (uint32_t)first);
    for (; i < e && X->occ_pos[i] <= (uint32_t)last; ++i)
        if (!tm_hidden(P, ex, X->occ_pos[i])) return 1;
    return 0;
}
TM_FN int tm_lit_true(const tm_params *P, const tm_corpus *X, const tm_example *ex, uint32_t lit) {
    return tm_positive_true(P, X, ex, tm_lit_seg(lit), tm_lit_sym(lit)) != (int)tm_lit_neg(lit);
}

/* ---- gates (spec 4 step 5; oracle _py_q32, _pc_q32) ---- */
TM_FN int32_t tm_clamp_t(const tm_params *P, int32_t s) { return s < -P->T ? -P->T : (s > P->T ? P->T : s); }
TM_FN uint32_t tm_py_q32(const tm_params *P, int32_t s_y) {
    uint64_t v = ((uint64_t)(uint32_t)(P->T - tm_clamp_t(P, s_y)) << 32) / (uint64_t)(2 * P->T);
    return v > TM_Q32_MAX ? TM_Q32_MAX : (uint32_t)v;
}
TM_FN uint32_t tm_pc_q32(const tm_params *P, int32_t s_c) {
    uint64_t v = ((uint64_t)(uint32_t)(P->T + tm_clamp_t(P, s_c)) << 32) / (uint64_t)(2 * P->T);
    return v > TM_Q32_MAX ? TM_Q32_MAX : (uint32_t)v;
}

/* ---- votes (oracle _vote_slot, _vote, _owns, _mass) ---- */
TM_FN int32_t tm_vote_slot(const tm_params *P, const tm_state *S, uint32_t j, uint32_t c) {
    const uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS;
    for (uint32_t k = 0; k < P->V_SLOTS; ++k) if (cls[k] == c) return (int32_t)k;
    return -1;
}
TM_FN int32_t tm_vote(const tm_params *P, const tm_state *S, uint32_t j, uint32_t c) {
    int32_t k = tm_vote_slot(P, S, j, c);
    return k < 0 ? 0 : S->vote_w[(uint64_t)j * P->V_SLOTS + k];
}
TM_FN int tm_owns(const tm_params *P, const tm_state *S, uint32_t j, uint32_t c) { return tm_vote(P, S, j, c) > 0; }
TM_FN int tm_owns_any(const tm_params *P, const tm_state *S, uint32_t j) {
    const uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS; const int16_t *w = S->vote_w + (uint64_t)j * P->V_SLOTS;
    for (uint32_t k = 0; k < P->V_SLOTS; ++k) if (cls[k] != TM_CLS_EMPTY && w[k] > 0) return 1;
    return 0;
}
TM_FN uint64_t tm_mass(const tm_params *P, const tm_state *S, uint32_t j) {
    const uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS; const int16_t *w = S->vote_w + (uint64_t)j * P->V_SLOTS;
    uint64_t m = 0;
    for (uint32_t k = 0; k < P->V_SLOTS; ++k) if (cls[k] != TM_CLS_EMPTY && w[k] > 0) m += S->G[cls[k]];
    return m;
}

/* ---- firing (oracle _fires against inc_lit, _fires_live against the slots, _fires_dense) ---- */
TM_FN int tm_fires(const tm_params *P, const tm_corpus *X, const tm_state *S, const tm_example *ex, uint32_t j) {
    if (!S->alive[j] || S->anchor[j] == TM_ANCHOR_NONE) return 0;
    const uint32_t *inc = S->inc_lit + (uint64_t)j * P->L_SLOTS;
    for (uint32_t k = 0; k < S->n_inc[j]; ++k) if (!tm_lit_true(P, X, ex, inc[k])) return 0;
    return 1;
}
TM_FN int tm_fires_live(const tm_params *P, const tm_corpus *X, const tm_state *S, const tm_example *ex, uint32_t j) {
    if (!S->alive[j]) return 0;
    const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS; const uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS;
    int has_anchor = 0;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) {
        if (st[k] >= TM_INCLUDE_THRESHOLD && ids[k] != TM_LIT_EMPTY) {
            if (!tm_lit_true(P, X, ex, ids[k])) return 0;
            if ((ids[k] & 1u) == 0 && (ids[k] >> 17) < TM_N_EXACT_SEG) has_anchor = 1;
        }
    }
    return has_anchor;
}
TM_FN int tm_fires_dense(const tm_params *P, const tm_corpus *X, const tm_state *S, const tm_example *ex, uint32_t j, int training) {
    const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS; const uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS;
    uint32_t n_included = 0;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) {
        if (st[k] >= TM_INCLUDE_THRESHOLD && ids[k] != TM_LIT_EMPTY) {
            if (!tm_lit_true(P, X, ex, ids[k])) return 0;
            ++n_included;
        }
    }
    return training || n_included > 0;
}

/* ---- literal state moves (oracle _state_up, _state_down) ---- */
TM_FN void tm_state_up(const tm_params *P, tm_state *S, uint32_t j, uint32_t k) {
    uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS + k;
    uint32_t s = *st;
    if (s == 255) return;
    if (s == TM_INCLUDE_THRESHOLD - 1) {
        uint32_t neg = S->lit_id[(uint64_t)j * P->L_SLOTS + k] & 1u;
        if (!P->dense_equivalent && S->n_inc[j] >= P->BUDGET) return;
        if (!P->dense_equivalent && neg && P->neg_budget >= 0 && (int32_t)S->n_inc_neg[j] >= P->neg_budget) return;
        S->n_inc[j] += 1;
        S->n_inc_neg[j] = (uint8_t)(S->n_inc_neg[j] + neg);
    }
    *st = (uint8_t)(s + 1);
}
TM_FN void tm_state_down(const tm_params *P, tm_state *S, uint32_t j, uint32_t k, uint32_t dec) {
    uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS + k;
    uint32_t s = *st, ns = dec >= s ? 0 : s - dec;
    if (s >= TM_INCLUDE_THRESHOLD && ns < TM_INCLUDE_THRESHOLD) {
        S->n_inc[j] -= 1;
        S->n_inc_neg[j] = (uint8_t)(S->n_inc_neg[j] - (S->lit_id[(uint64_t)j * P->L_SLOTS + k] & 1u));
    }
    *st = (uint8_t)ns;
    if (ns == 0) S->lit_id[(uint64_t)j * P->L_SLOTS + k] = TM_LIT_EMPTY;
}

/* ---- settle (spec 5.4; oracle _settle) and the ownership-change mark (_mark) ---- */
TM_FN void tm_settle(const tm_params *P, tm_state *S, uint32_t j) {
    if (!(P->decay_mode == TM_DECAY_LAZY_OWNED && !P->dense_equivalent)) return;
    uint64_t m = tm_mass(P, S, j);
    uint64_t sub = S->decay_mark[j] + S->own_fire_mass[j];
    uint64_t owed = m > sub ? m - sub : 0;
    uint64_t n_q16 = (uint64_t)(((unsigned __int128)owed * (unsigned __int128)P->s_den) / (unsigned __int128)P->s_num);
    uint64_t n_int = n_q16 >> 16;
    uint32_t frac_q32 = (uint32_t)((n_q16 & 0xFFFFull) << 16);
    if (n_q16 > 0) {
        const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS;
        for (uint32_t k = 0; k < P->L_SLOTS; ++k) {
            if (ids[k] == TM_LIT_EMPTY) continue;
            uint64_t dec = n_int + (tm_bern(frac_q32, P->seed, TM_S_DECAY, S->step, j, k) ? 1 : 0);
            if (dec) tm_state_down(P, S, j, k, dec > 255 ? 255 : (uint32_t)dec);
        }
    }
    S->decay_mark[j] = m;
    S->own_fire_mass[j] = 0;
}
TM_FN void tm_mark(const tm_params *P, tm_state *S, uint32_t j) {
    if (P->decay_mode == TM_DECAY_LAZY_OWNED && !P->dense_equivalent) S->decay_mark[j] = tm_mass(P, S, j);
}

/* ---- vote_add (spec 5.2; oracle _vote_add). Returns 1 iff a slot for c exists or was obtained. ---- */
TM_FN int32_t tm_lowest_free_vote_slot(const tm_params *P, const tm_state *S, uint32_t j) {
    const uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS;
    for (uint32_t k = 0; k < P->V_SLOTS; ++k) if (cls[k] == TM_CLS_EMPTY) return (int32_t)k;
    return -1;
}
TM_FN int tm_vote_add(const tm_params *P, tm_state *S, uint32_t j, uint32_t c, int32_t delta) {
    uint16_t *cls = S->vote_cls + (uint64_t)j * P->V_SLOTS; int16_t *w = S->vote_w + (uint64_t)j * P->V_SLOTS;
    int32_t k = tm_vote_slot(P, S, j, c);
    if (k >= 0) {
        int32_t nw = (int32_t)w[k] + delta;
        if (nw < -TM_W_MAX) nw = -TM_W_MAX;
        if (nw > TM_W_MAX) nw = TM_W_MAX;
        int change = (w[k] > 0) != (nw > 0);
        if (change) tm_settle(P, S, j);
        w[k] = (int16_t)nw;
        if (nw == 0) cls[k] = TM_CLS_EMPTY;
        if (change) tm_mark(P, S, j);
        return 1;
    }
    k = tm_lowest_free_vote_slot(P, S, j);
    if (k < 0) {
        /* the slot with the smallest |w|, ties toward the lower index; replace only if |w| <= 1 */
        int32_t best = 0; int32_t best_abs = w[0] < 0 ? -w[0] : w[0];
        for (uint32_t q = 1; q < P->V_SLOTS; ++q) { int32_t aw = w[q] < 0 ? -w[q] : w[q]; if (aw < best_abs) { best_abs = aw; best = (int32_t)q; } }
        if (best_abs > 1) return 0;
        k = best;
    }
    int change = delta > 0 || w[k] > 0;
    if (change) tm_settle(P, S, j);
    cls[k] = (uint16_t)c; w[k] = (int16_t)delta;
    if (change) tm_mark(P, S, j);
    return 1;
}

/* ---- insertion slot rule (spec 5.2.1; oracle _insert_literal) ---- */
TM_FN int tm_insert_literal(const tm_params *P, tm_state *S, uint32_t j, uint32_t lit) {
    uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS; uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) if (ids[k] == lit) return 0;
    int32_t slot = -1;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) if (ids[k] == TM_LIT_EMPTY) { slot = (int32_t)k; break; }
    if (slot < 0) {
        uint32_t best_state = 256;
        for (uint32_t k = 0; k < P->L_SLOTS; ++k) if (st[k] < TM_INCLUDE_THRESHOLD && st[k] < best_state) { best_state = st[k]; slot = (int32_t)k; }
        if (slot < 0 || best_state >= TM_STATE_INSERT) return 0;
    }
    ids[slot] = lit; st[slot] = TM_STATE_INSERT;
    return 1;
}

/* ---- candidate sampling (spec 5.2.3; oracle _sample_candidate). Returns lit or TM_LIT_EMPTY. ---- */
TM_FN uint32_t tm_first_cum_gt(const uint32_t *cum, uint32_t n, uint32_t x) {
    uint32_t i = 0; while (i < n && cum[i] <= x) ++i; return i;   /* bisect_right */
}
TM_FN uint32_t tm_freq(const tm_params *P, const tm_corpus *X, uint32_t sym);
TM_FN uint32_t tm_sample_candidate_once(const tm_params *P, const tm_corpus *X, const tm_example *ex,
                                        uint32_t stream, uint32_t j, uint32_t t, int exact_only);
TM_FN uint32_t tm_sample_candidate(const tm_params *P, const tm_corpus *X, const tm_example *ex,
                                   uint32_t stream, uint32_t j, uint32_t t, int exact_only) {
    uint32_t best = TM_LIT_EMPTY, best_key = 0; int have = 0;
    for (uint32_t i = 0; i < P->cand_tries; ++i) {
        uint32_t lit = tm_sample_candidate_once(P, X, ex, stream, j, t + 9 * i, exact_only);
        if (lit == TM_LIT_EMPTY) continue;
        uint32_t key = tm_freq(P, X, tm_lit_sym(lit));
        if (!have || key < best_key) { have = 1; best = lit; best_key = key; }
    }
    return best;
}
TM_FN uint32_t tm_sample_candidate_once(const tm_params *P, const tm_corpus *X, const tm_example *ex,
                                        uint32_t stream, uint32_t j, uint32_t t, int exact_only) {
    const uint32_t *seg_ids = exact_only ? P->exact_seg_ids : P->seg_ids;
    const uint32_t *cum = exact_only ? P->exact_cum : P->seg_cum;
    uint32_t n = exact_only ? P->n_exact : P->n_segs, total = exact_only ? P->exact_total : P->seg_total;
    uint32_t seg = seg_ids[tm_first_cum_gt(cum, n, tm_pick(total, P->seed, stream, ex->a, j, t))];
    if (P->mode == TM_MODE_TABULAR) {                                 /* N8 */
        uint32_t f = tm_pick(P->n_features, P->seed, stream, ex->a, j, t + 2);
        return ex->x[f] == 1 ? tm_lit_pack(seg, f, 0) : TM_LIT_EMPTY;
    }
    uint32_t u = tm_rng32(P->seed, stream, ex->a, j, t + 1);
    if (seg < TM_N_EXACT_SEG) {
        uint32_t kind = u < P->kind_t70 ? 0 : u < P->kind_t80 ? 1 : u < P->kind_t90 ? 2 : 3;
        uint32_t sym = ex->ctx[seg][kind];
        return sym == P->sym_mask ? TM_LIT_EMPTY : tm_lit_pack(seg, sym, 0);
    }
    uint32_t kind = u < P->kind_t70 ? 0 : 2;
    uint32_t k = tm_bag_bucket(seg), side = tm_bag_side(seg);
    uint32_t lo = tm_bag_lo(k), hi = tm_bag_hi(k);
    uint32_t dist = lo + tm_pick(hi - lo + 1, P->seed, stream, ex->a, j, t + 2);
    int64_t q = side == 0 ? (int64_t)ex->p - (int64_t)dist : (int64_t)ex->p + (int64_t)dist;
    if (q < (int64_t)ex->doc_lo || q >= (int64_t)ex->doc_hi || tm_hidden(P, ex, (uint32_t)q)) return TM_LIT_EMPTY;
    uint32_t tok = X->tok[q];
    if (X->stop[tok] != 0) return TM_LIT_EMPTY;
    return tm_lit_pack(seg, X->kind_map[kind * P->V + tok], 0);
}

/* ---- Type Ia, Ib, II (spec 5.2.1, 5.2.2; oracle _type_ia, _type_ib, _type_ii and dense forms) ---- */
TM_FN void tm_type_ia(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j, int with_insertion, uint64_t *diag) {
    diag[TM_D_N_TYPE1] += 1;
    const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) {
        if (ids[k] == TM_LIT_EMPTY) continue;
        if (tm_lit_true(P, X, ex, ids[k])) tm_state_up(P, S, j, k);
        else if (tm_bern(P->inv_s_q32, P->seed, TM_S_TYPE1_DEC, ex->a, j, k)) tm_state_down(P, S, j, k, 1);
    }
    if (with_insertion && tm_bern(P->p_ins_q32, P->seed, TM_S_INSERT, ex->a, j, 0)) {
        uint32_t lit = tm_sample_candidate(P, X, ex, TM_S_INSERT_PICK, j, 0, 0);
        if (lit != TM_LIT_EMPTY && tm_insert_literal(P, S, j, lit)) diag[TM_D_N_INSERTED] += 1;
    }
}
TM_FN void tm_type_ib(const tm_params *P, tm_state *S, const tm_example *ex, uint32_t j, uint64_t *diag) {
    diag[TM_D_N_TYPE1] += 1;
    const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k)
        if (ids[k] != TM_LIT_EMPTY && tm_bern(P->inv_s_q32, P->seed, TM_S_TYPE1_DEC, ex->a, j, k)) tm_state_down(P, S, j, k, 1);
}
TM_FN void tm_type_ii(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j, int with_neg_insertion, uint64_t *diag) {
    diag[TM_D_N_TYPE2] += 1;
    const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS; const uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS;
    for (uint32_t k = 0; k < P->L_SLOTS; ++k)
        if (ids[k] != TM_LIT_EMPTY && st[k] < TM_INCLUDE_THRESHOLD && !tm_lit_true(P, X, ex, ids[k])) tm_state_up(P, S, j, k);
    if (with_neg_insertion && tm_bern(P->p_neg_q32, P->seed, TM_S_NEG_INSERT, ex->a, j, 0)) {
        uint32_t lit = tm_sample_candidate(P, X, ex, TM_S_NEG_PICK, j, 0, 1);
        if (lit != TM_LIT_EMPTY && tm_insert_literal(P, S, j, lit | 1u)) diag[TM_D_N_NEG_INSERTED] += 1;
    }
}

/* ---- fire statistics (oracle _fire_stats) ---- */
TM_FN void tm_fire_stats(const tm_params *P, tm_state *S, const tm_example *ex, uint32_t j) {
    if (S->stat_fire[j] != TM_U32_MAX) S->stat_fire[j] += 1;
    if (tm_owns(P, S, j, ex->y)) {
        if (S->stat_tp[j] != TM_U32_MAX) S->stat_tp[j] += 1;
        S->last_tp_step[j] = (uint32_t)(S->step & TM_U32_MAX);
        if (P->decay_mode == TM_DECAY_LAZY_OWNED && !P->dense_equivalent) S->own_fire_mass[j] += (uint64_t)(ex->py_q32 >> 16);
    }
}

/* ---- target feedback (spec 5.2; oracle _target_feedback) ---- */
TM_FN void tm_target_feedback(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j, uint64_t *diag) {
    if (!tm_bern(ex->py_q32, P->seed, TM_S_GATE_TGT, ex->a, j, 0)) return;
    diag[TM_D_N_TGT_GATE_PASS] += 1;
    int32_t w = tm_vote(P, S, j, ex->y);
    if (w > 0) { tm_type_ia(P, X, S, ex, j, 1, diag); tm_vote_add(P, S, j, ex->y, +1); }
    else if (w < 0) { tm_type_ii(P, X, S, ex, j, 1, diag); tm_vote_add(P, S, j, ex->y, +1); }
    else if (tm_vote_add(P, S, j, ex->y, +1)) tm_type_ia(P, X, S, ex, j, 1, diag);
}

/* ---- negatives (spec 5.3; oracle _negative_feedback, _negative_update) ---- */
TM_FN void tm_negative_update(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j,
                              uint32_t c, int32_t s_c, uint32_t n, uint64_t *diag) {
    diag[TM_D_N_NEG_GATE_PASS] += 1;
    int32_t w = tm_vote(P, S, j, c);
    if (w > 0) { tm_type_ii(P, X, S, ex, j, 1, diag); tm_vote_add(P, S, j, c, -1); }
    else if (w < 0) { tm_type_ia(P, X, S, ex, j, 1, diag); tm_vote_add(P, S, j, c, -1); }
    else if (s_c >= ex->s_y - P->MARGIN) {
        if (tm_bern(P->p_veto_q32, P->seed, TM_S_VETO, ex->a, j, n)) {
            int32_t k = tm_lowest_free_vote_slot(P, S, j);
            if (k >= 0) { S->vote_cls[(uint64_t)j * P->V_SLOTS + k] = (uint16_t)c; S->vote_w[(uint64_t)j * P->V_SLOTS + k] = -1; diag[TM_D_N_VETOES] += 1; }
        }
    }
}
/* scores_of(c): the batch-start score S_i[c] of the example; needed only in uniform_binomial mode.
 * The orchestration passes the example's score lookup through `scores` (dense int32[V] row) or 0. */
TM_FN void tm_negative_feedback(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j,
                                const int32_t *scores, uint64_t *diag) {
    if (P->neg_mode == TM_NEG_FOCUSED && !P->dense_equivalent) {
        for (uint32_t n = 0; n < ex->n_neg; ++n)
            if (tm_bern(ex->neg_pc_q32[n], P->seed, TM_S_GATE_NEG, ex->a, j, n))
                tm_negative_update(P, X, S, ex, j, ex->neg_cls[n], ex->neg_s[n], n, diag);
        return;
    }
    if (P->neg_mode == TM_NEG_BINOMIAL && !P->dense_equivalent) {          /* N14: exact Binomial draw */
        uint32_t u = tm_rng32(P->seed, TM_S_NEG_DRAW, ex->a, j, 0);
        uint32_t k = 0;
        while (k < TM_NEG_K_MAX && P->neg_cdf_q32[k] <= u) ++k;
        uint32_t chosen[TM_NEG_K_MAX]; uint32_t n = 0, t = 0;
        while (n < k && t < 8 * TM_NEG_K_MAX) {
            uint32_t c = tm_pick(P->V - 1, P->seed, TM_S_NEG_DRAW, ex->a, j, 1 + t);
            ++t;
            if (c >= ex->y) ++c;
            int dup = 0;
            for (uint32_t i = 0; i < n; ++i) if (chosen[i] == c) { dup = 1; break; }
            if (!dup) chosen[n++] = c;
        }
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t c = chosen[i];
            int32_t s_c = scores ? scores[c] : 0;
            if (tm_bern(tm_pc_q32(P, s_c), P->seed, TM_S_GATE_NEG, ex->a, j, c))
                tm_negative_update(P, X, S, ex, j, c, s_c, c, diag);
        }
        return;
    }
    for (uint32_t c = 0; c < P->V; ++c) {
        if (c == ex->y) continue;
        if (!tm_bern(P->q_neg_q32, P->seed, TM_S_GATE_NEG, ex->a, j, TM_NEG_CLASS_DRAW_BASE + c)) continue;
        int32_t s_c = scores ? scores[c] : 0;
        if (tm_bern(tm_pc_q32(P, s_c), P->seed, TM_S_GATE_NEG, ex->a, j, c))
            tm_negative_update(P, X, S, ex, j, c, s_c, c, diag);              /* N5 */
    }
}

/* ---- one event (spec 5.1; oracle _event) ---- */
TM_FN void tm_event(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j,
                    const int32_t *scores, uint64_t *diag) {
    if (!tm_fires_live(P, X, S, ex, j)) { diag[TM_D_N_EVENTS_SKIPPED_NOT_FIRING] += 1; return; }
    diag[TM_D_N_EVENTS_APPLIED] += 1;
    tm_fire_stats(P, S, ex, j);
    tm_target_feedback(P, X, S, ex, j, diag);          /* N3 */
    tm_negative_feedback(P, X, S, ex, j, scores, diag);
}

/* ---- seeding of one clause (spec 5.6; oracle _seed_literals and the body of _seed) ---- */
TM_FN uint32_t tm_seed_literals(const tm_params *P, const tm_corpus *X, const tm_example *ex, uint32_t *lits) {
    uint32_t u = tm_rng32(P->seed, TM_S_SEED_PICK, ex->a, 0, 100);
    uint32_t n_lit = u < P->nlit_t1 ? 1 : u < P->nlit_t2 ? 2 : 3;
    uint32_t n = 0;
    for (uint32_t m = 0; m < n_lit; ++m) {
        uint32_t lit = tm_sample_candidate(P, X, ex, TM_S_SEED_PICK, 0, 3 * m, 1);
        if (lit == TM_LIT_EMPTY) continue;
        int dup = 0;
        for (uint32_t i = 0; i < n; ++i) if ((lit >> 17) == (lits[i] >> 17)) { dup = 1; break; }
        if (dup) continue;
        lits[n++] = lit;
    }
    return n;
}
TM_FN void tm_seed_clause(const tm_params *P, tm_state *S, const tm_example *ex, uint32_t j, const uint32_t *lits, uint32_t n) {
    for (uint32_t k = 0; k < n; ++k) { S->lit_id[(uint64_t)j * P->L_SLOTS + k] = lits[k]; S->lit_state[(uint64_t)j * P->L_SLOTS + k] = TM_STATE_SEED; }
    S->vote_cls[(uint64_t)j * P->V_SLOTS] = (uint16_t)ex->y; S->vote_w[(uint64_t)j * P->V_SLOTS] = 1;
    S->alive[j] = 1;
    S->birth_step[j] = S->last_tp_step[j] = (uint32_t)(S->step & TM_U32_MAX);
    S->stat_fire[j] = S->stat_tp[j] = 0;
    S->decay_mark[j] = (P->decay_mode == TM_DECAY_LAZY_OWNED && !P->dense_equivalent) ? S->G[ex->y] : 0;
    S->own_fire_mass[j] = 0;
}

/* ---- recycle (spec 5.7; oracle _recycle, without the free-list push, which is orchestration) ---- */
TM_FN void tm_recycle(const tm_params *P, tm_state *S, uint32_t j) {
    for (uint32_t k = 0; k < P->L_SLOTS; ++k) { S->lit_id[(uint64_t)j * P->L_SLOTS + k] = TM_LIT_EMPTY; S->lit_state[(uint64_t)j * P->L_SLOTS + k] = 0; }
    for (uint32_t k = 0; k < P->V_SLOTS; ++k) { S->vote_cls[(uint64_t)j * P->V_SLOTS + k] = TM_CLS_EMPTY; S->vote_w[(uint64_t)j * P->V_SLOTS + k] = 0; }
    S->decay_mark[j] = S->own_fire_mass[j] = 0;
    S->stat_fire[j] = S->stat_tp[j] = 0;
    S->birth_step[j] = S->last_tp_step[j] = 0;
    S->alive[j] = 0;
}

/* ---- refresh of one clause (spec 2; oracle _refresh without the index update). Returns the new anchor. ---- */
TM_FN uint32_t tm_freq(const tm_params *P, const tm_corpus *X, uint32_t sym) {
    if (P->mode == TM_MODE_TABULAR) return 0;                          /* N8 */
    if (sym == P->sym_mask || sym == P->sym_edge) return TM_U32_MAX;
    return X->sym_freq[sym];
}
TM_FN void tm_sort_u32(uint32_t *a, uint32_t n) {   /* insertion sort, n <= 64 */
    for (uint32_t i = 1; i < n; ++i) { uint32_t v = a[i]; uint32_t k = i; while (k > 0 && a[k - 1] > v) { a[k] = a[k - 1]; --k; } a[k] = v; }
}
TM_FN uint32_t tm_refresh_clause(const tm_params *P, const tm_corpus *X, tm_state *S, uint32_t j) {
    uint32_t groups[4][TM_MAX_L_SLOTS]; uint32_t gn[4] = {0, 0, 0, 0};
    if (S->alive[j]) {
        const uint32_t *ids = S->lit_id + (uint64_t)j * P->L_SLOTS; const uint8_t *st = S->lit_state + (uint64_t)j * P->L_SLOTS;
        for (uint32_t k = 0; k < P->L_SLOTS; ++k) {
            if (st[k] >= TM_INCLUDE_THRESHOLD && ids[k] != TM_LIT_EMPTY) {
                uint32_t g = ((ids[k] >> 17) >= TM_N_EXACT_SEG ? 2 : 0) + (ids[k] & 1u);
                groups[g][gn[g]++] = ids[k];
            }
        }
    }
    uint32_t *inc = S->inc_lit + (uint64_t)j * P->L_SLOTS; uint32_t n = 0;
    for (uint32_t g = 0; g < 4; ++g) { tm_sort_u32(groups[g], gn[g]); for (uint32_t i = 0; i < gn[g]; ++i) inc[n++] = groups[g][i]; }
    S->n_inc[j] = (uint8_t)n;
    S->n_inc_neg[j] = (uint8_t)(gn[1] + gn[3]);
    /* anchor: smallest sym_freq among positive exact literals, ties toward the lower literal id */
    uint32_t best = TM_ANCHOR_NONE; uint32_t best_freq = 0; int have = 0;
    for (uint32_t i = 0; i < gn[0]; ++i) {
        uint32_t lit = groups[0][i], f = tm_freq(P, X, tm_lit_sym(lit));
        if (!have || f < best_freq || (f == best_freq && lit < best)) { have = 1; best_freq = f; best = lit; }
    }
    return best;
}

/* ---- dense-equivalent feedback (spec 6; oracle _dense_clause_feedback, _dense_update) ---- */
TM_FN void tm_dense_update(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j,
                           uint32_t c, int32_t delta, int fires, uint64_t *diag) {
    int32_t k = tm_vote_slot(P, S, j, c);
    int16_t *w = S->vote_w + (uint64_t)j * P->V_SLOTS + k;
    int type_i = (*w >= 0) == (delta > 0);
    if (type_i && fires) tm_type_ia(P, X, S, ex, j, 0, diag);
    else if (type_i) tm_type_ib(P, S, ex, j, diag);
    else if (fires) tm_type_ii(P, X, S, ex, j, 0, diag);
    if (fires) {
        int32_t nw = (int32_t)*w + delta;
        if (nw < -TM_W_MAX) nw = -TM_W_MAX;
        if (nw > TM_W_MAX) nw = TM_W_MAX;
        *w = (int16_t)nw;
    }
}
TM_FN void tm_dense_clause_feedback(const tm_params *P, const tm_corpus *X, tm_state *S, const tm_example *ex, uint32_t j,
                                    int fires, const int32_t *scores, uint64_t *diag) {
    if (fires) { diag[TM_D_N_EVENTS_APPLIED] += 1; tm_fire_stats(P, S, ex, j); }
    if (tm_bern(ex->py_q32, P->seed, TM_S_GATE_TGT, ex->a, j, 0)) { diag[TM_D_N_TGT_GATE_PASS] += 1; tm_dense_update(P, X, S, ex, j, ex->y, +1, fires, diag); }
    for (uint32_t c = 0; c < P->V; ++c) {
        if (c == ex->y) continue;
        if (!tm_bern(P->q_neg_q32, P->seed, TM_S_GATE_NEG, ex->a, j, TM_NEG_CLASS_DRAW_BASE + c)) continue;
        if (tm_bern(tm_pc_q32(P, scores[c]), P->seed, TM_S_GATE_NEG, ex->a, j, c)) { diag[TM_D_N_NEG_GATE_PASS] += 1; tm_dense_update(P, X, S, ex, j, c, -1, fires, diag); }
    }
}

#endif /* TMLM_TM_CORE_H */
