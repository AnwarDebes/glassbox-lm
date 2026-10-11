/* engine.cpp: the C++17 + OpenMP host engine behind a plain C ABI (docs/03 "Implementations",
 * docs/04 M3). All learning logic is in ../core/tm_core.h; this file is the batch orchestration
 * of docs/11 sections 4 and 5 in the normative order, plus the inverted index, the free list,
 * state import/export and the C entry points used by cpu_engine.py through ctypes.
 *
 * Determinism: every random draw is a pure function of (seed, stream, example uid, clause,
 * slot); clauses are independent during feedback, so the per-clause loop is parallel and the
 * result cannot depend on the thread count. Everything that depends on order (seeding, the
 * sweep's top-up, the free list) is sequential and mirrors the oracle exactly.
 */
#include "../core/tm_core.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Index {
    /* anchor literal -> ascending clause ids (oracle: self.index) */
    std::unordered_map<uint32_t, std::vector<uint32_t>> post;
    void remove(uint32_t lit, uint32_t j) {
        auto it = post.find(lit);
        if (it == post.end()) return;
        auto &v = it->second;
        auto pos = std::lower_bound(v.begin(), v.end(), j);
        if (pos != v.end() && *pos == j) v.erase(pos);
        if (v.empty()) post.erase(it);
    }
    void insert(uint32_t lit, uint32_t j) {
        auto &v = post[lit];
        v.insert(std::upper_bound(v.begin(), v.end(), j), j);   /* insort: keeps ascending order */
    }
    const std::vector<uint32_t> *get(uint32_t lit) const {
        auto it = post.find(lit);
        return it == post.end() ? nullptr : &it->second;
    }
};

struct Engine {
    tm_params P{};
    tm_corpus X{};
    tm_state S{};
    /* owned state storage */
    std::vector<uint32_t> lit_id; std::vector<uint8_t> lit_state;
    std::vector<uint16_t> vote_cls; std::vector<int16_t> vote_w;
    std::vector<uint64_t> decay_mark, own_fire_mass, G;
    std::vector<uint32_t> stat_fire, stat_tp, birth_step, last_tp_step;
    std::vector<uint8_t> alive;
    std::vector<uint32_t> inc_lit; std::vector<uint8_t> n_inc, n_inc_neg; std::vector<uint32_t> anchor;
    Index index;
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> free_heap;
    uint64_t diag[TM_D_COUNT];
    int is_dense = 0, is_tabular = 0;

    void alloc() {
        uint64_t C = P.C;
        lit_id.assign(C * P.L_SLOTS, TM_LIT_EMPTY); lit_state.assign(C * P.L_SLOTS, 0);
        vote_cls.assign(C * P.V_SLOTS, TM_CLS_EMPTY); vote_w.assign(C * P.V_SLOTS, 0);
        decay_mark.assign(C, 0); own_fire_mass.assign(C, 0); G.assign(P.V, 0);
        stat_fire.assign(C, 0); stat_tp.assign(C, 0); birth_step.assign(C, 0); last_tp_step.assign(C, 0);
        alive.assign(C, 0);
        inc_lit.assign(C * P.L_SLOTS, 0); n_inc.assign(C, 0); n_inc_neg.assign(C, 0); anchor.assign(C, TM_ANCHOR_NONE);
        S.lit_id = lit_id.data(); S.lit_state = lit_state.data(); S.vote_cls = vote_cls.data(); S.vote_w = vote_w.data();
        S.decay_mark = decay_mark.data(); S.own_fire_mass = own_fire_mass.data(); S.G = G.data();
        S.stat_fire = stat_fire.data(); S.stat_tp = stat_tp.data(); S.birth_step = birth_step.data(); S.last_tp_step = last_tp_step.data();
        S.alive = alive.data(); S.step = 0;
        S.inc_lit = inc_lit.data(); S.n_inc = n_inc.data(); S.n_inc_neg = n_inc_neg.data(); S.anchor = anchor.data();
    }

    void init_dense() {   /* spec 6 (oracle _init_dense) */
        for (uint32_t j = 0; j < P.C; ++j) {
            for (uint32_t f = 0; f < P.n_features; ++f) {
                lit_id[(uint64_t)j * P.L_SLOTS + 2 * f] = tm_lit_pack(0, f, 0);
                lit_id[(uint64_t)j * P.L_SLOTS + 2 * f + 1] = tm_lit_pack(0, f, 1);
                lit_state[(uint64_t)j * P.L_SLOTS + 2 * f] = TM_STATE_DENSE_INIT;
                lit_state[(uint64_t)j * P.L_SLOTS + 2 * f + 1] = TM_STATE_DENSE_INIT;
            }
            for (uint32_t c = 0; c < P.V; ++c) {
                vote_cls[(uint64_t)j * P.V_SLOTS + c] = (uint16_t)c;
                vote_w[(uint64_t)j * P.V_SLOTS + c] = (tm_rng32(P.seed, TM_S_SEED, j, c, 0) & 1u) ? 1 : -1;
            }
            alive[j] = 1;
        }
    }

    /* refresh(j) with index maintenance (oracle _refresh) */
    void refresh(uint32_t j) {
        uint32_t new_anchor = tm_refresh_clause(&P, &X, &S, j);
        uint32_t old_anchor = anchor[j];
        if (new_anchor != old_anchor) {
            if (old_anchor != TM_ANCHOR_NONE) index.remove(old_anchor, j);
            if (new_anchor != TM_ANCHOR_NONE) index.insert(new_anchor, j);
            anchor[j] = new_anchor;
        }
    }
    void rebuild_derived() {   /* oracle _rebuild_derived */
        index.post.clear();
        std::fill(anchor.begin(), anchor.end(), TM_ANCHOR_NONE);
        for (uint32_t j = 0; j < P.C; ++j) refresh(j);
        free_heap = {};
        for (uint32_t j = 0; j < P.C; ++j) if (!alive[j]) free_heap.push(j);
    }

    /* ---------------- forward pass (spec 4) ---------------- */
    struct Fwd {
        std::vector<uint32_t> fired;        /* F(i), ascending */
        std::vector<int32_t> scores;        /* dense [V] (only touched entries non-zero) */
        std::vector<uint32_t> touched;      /* classes with a non-zero entry */
        uint32_t n_candidates = 0;
    };

    void candidates(const tm_example &ex, std::vector<uint32_t> &out) const {   /* oracle _candidates */
        out.clear();
        if (is_tabular) {
            for (uint32_t f = 0; f < P.n_features; ++f)
                if (ex.x[f] == 1) if (auto *v = index.get(tm_lit_pack(0, f, 0))) out.insert(out.end(), v->begin(), v->end());
            return;
        }
        for (uint32_t i = 0; i < P.n_exact; ++i) {
            uint32_t seg = P.exact_seg_ids[i];
            const uint16_t *row = ex.ctx[seg];
            uint32_t nsym = (row[0] == P.sym_mask || row[0] == P.sym_edge) ? 1 : 4;
            for (uint32_t k = 0; k < nsym; ++k)
                if (auto *v = index.get(tm_lit_pack(seg, row[k], 0))) out.insert(out.end(), v->begin(), v->end());
        }
    }

    void forward(tm_example &ex, Fwd &fw, std::vector<uint32_t> &scratch, int training) {
        /* fired (oracle _fired) */
        fw.fired.clear();
        if (is_dense) {
            fw.n_candidates = P.C;
            for (uint32_t j = 0; j < P.C; ++j) if (tm_fires_dense(&P, &X, &S, &ex, j, training)) fw.fired.push_back(j);
        } else {
            candidates(ex, scratch);
            fw.n_candidates = (uint32_t)scratch.size();
            for (uint32_t j : scratch) if (tm_fires(&P, &X, &S, &ex, j)) fw.fired.push_back(j);
            std::sort(fw.fired.begin(), fw.fired.end());
        }
        /* scores (oracle _scores) */
        for (uint32_t c : fw.touched) fw.scores[c] = 0;
        fw.touched.clear();
        for (uint32_t j : fw.fired) {
            const uint16_t *cls = vote_cls.data() + (uint64_t)j * P.V_SLOTS; const int16_t *w = vote_w.data() + (uint64_t)j * P.V_SLOTS;
            for (uint32_t k = 0; k < P.V_SLOTS; ++k) if (cls[k] != TM_CLS_EMPTY) {
                if (fw.scores[cls[k]] == 0) fw.touched.push_back(cls[k]);
                fw.scores[cls[k]] += w[k];
            }
        }
        /* a class may be pushed more than once when its sum passes through 0: make `touched` unique.
         * A touched class whose sum is exactly 0 stays listed (the oracle's dict keeps it too). */
        std::sort(fw.touched.begin(), fw.touched.end());
        fw.touched.erase(std::unique(fw.touched.begin(), fw.touched.end()), fw.touched.end());
        if (!training) return;
        ex.s_y = fw.scores[ex.y];
        ex.py_q32 = tm_py_q32(&P, ex.s_y);
        ex.cover = 0;
        for (uint32_t j : fw.fired) if (tm_owns(&P, &S, j, ex.y)) ex.cover += 1;
        /* negatives: up to K_NEG classes c != y with the largest S > 0, ties toward the lower class id */
        std::vector<std::pair<int32_t, uint32_t>> ranked;
        for (uint32_t c : fw.touched) if (c != ex.y && fw.scores[c] > 0) ranked.emplace_back(-fw.scores[c], c);
        std::sort(ranked.begin(), ranked.end());
        ex.n_neg = 0;
        for (size_t n = 0; n < ranked.size() && n < P.K_NEG; ++n) {
            ex.neg_cls[ex.n_neg] = (uint16_t)ranked[n].second;
            ex.neg_s[ex.n_neg] = -ranked[n].first;
            ex.neg_pc_q32[ex.n_neg] = tm_pc_q32(&P, -ranked[n].first);
            ex.n_neg += 1;
        }
    }

    /* ---------------- training (spec 5) ---------------- */
    void train(uint32_t epoch, const std::vector<uint32_t> &ids) {
        std::memset(diag, 0, sizeof diag);
        const size_t B = ids.size();
        std::vector<tm_example> ex(B);
        std::vector<Fwd> fw(B);
        for (size_t i = 0; i < B; ++i) { tm_make_example(&P, &X, &ex[i], epoch, ids[i], nullptr); fw[i].scores.assign(P.V, 0); }
        diag[TM_D_N_EXAMPLES] += B;
        if (is_dense) train_dense(ex[0], fw[0]);
        else train_sparse(ex, fw);
        S.step += 1;
    }

    void train_sparse(std::vector<tm_example> &ex, std::vector<Fwd> &fw) {
        const size_t B = ex.size();
        /* spec 4: forward pass of the whole batch against the state at batch start (parallel over examples) */
        #pragma omp parallel
        {
            std::vector<uint32_t> scratch;
            #pragma omp for schedule(dynamic, 16)
            for (size_t i = 0; i < B; ++i) forward(ex[i], fw[i], scratch, 1);
        }
        std::unordered_map<uint32_t, std::vector<uint32_t>> events;   /* touched clause -> example indices ascending */
        for (size_t i = 0; i < B; ++i) {
            diag[TM_D_N_CANDIDATES] += fw[i].n_candidates;
            diag[TM_D_N_FIRED] += fw[i].fired.size();
            for (uint32_t j : fw[i].fired) events[j].push_back((uint32_t)i);
        }
        std::vector<uint32_t> touched_clauses;
        touched_clauses.reserve(events.size());
        for (auto &kv : events) touched_clauses.push_back(kv.first);
        std::sort(touched_clauses.begin(), touched_clauses.end());
        std::vector<uint32_t> changed;
        /* spec 5.1 to 5.4: feedback, one clause at a time; clauses do not interact, so parallel */
        if (P.decay_mode == TM_DECAY_LAZY_OWNED) {
            changed = touched_clauses;
            #pragma omp parallel
            {
                uint64_t d[TM_D_COUNT]; std::memset(d, 0, sizeof d);
                #pragma omp for schedule(dynamic, 4)
                for (size_t t = 0; t < touched_clauses.size(); ++t) {
                    uint32_t j = touched_clauses[t];
                    const auto &lst = events[j];
                    tm_settle(&P, &S, j);
                    size_t cap = std::min<size_t>(lst.size(), P.E_CAP);
                    for (size_t n = 0; n < cap; ++n) tm_event(&P, &X, &S, &ex[lst[n]], j, fw[lst[n]].scores.data(), d);
                    d[TM_D_N_EVENTS_DROPPED_CAP] += lst.size() > P.E_CAP ? lst.size() - P.E_CAP : 0;
                }
                #pragma omp critical
                for (int q = 0; q < TM_D_COUNT; ++q) diag[q] += d[q];
            }
        } else {   /* N10: eager_exact in the sparse engine */
            for (uint32_t j = 0; j < P.C; ++j) if (alive[j]) changed.push_back(j);
            #pragma omp parallel
            {
                uint64_t d[TM_D_COUNT]; std::memset(d, 0, sizeof d);
                #pragma omp for schedule(dynamic, 4)
                for (size_t t = 0; t < changed.size(); ++t) {
                    uint32_t j = changed[t];
                    auto it = events.find(j);
                    const std::vector<uint32_t> *lst = it == events.end() ? nullptr : &it->second;
                    size_t place = 0;
                    for (size_t i = 0; i < B; ++i) {
                        int in_list = lst && place < lst->size() && (*lst)[place] == i;
                        if (!in_list) {
                            if (tm_owns(&P, &S, j, ex[i].y) && tm_bern(ex[i].py_q32, P.seed, TM_S_GATE_TGT, ex[i].a, j, 0)) {
                                d[TM_D_N_TGT_GATE_PASS] += 1; tm_type_ib(&P, &S, &ex[i], j, d);
                            }
                        } else {
                            if (place < P.E_CAP) tm_event(&P, &X, &S, &ex[i], j, fw[i].scores.data(), d);
                            else d[TM_D_N_EVENTS_DROPPED_CAP] += 1;
                            ++place;
                        }
                    }
                }
                #pragma omp critical
                for (int q = 0; q < TM_D_COUNT; ++q) diag[q] += d[q];
            }
        }
        /* spec 5.5 gate mass */
        for (size_t i = 0; i < B; ++i) G[ex[i].y] += (uint64_t)(ex[i].py_q32 >> 16);
        /* spec 5.6 seeding, sequential in ascending i */
        std::unordered_map<uint32_t, uint32_t> served;
        for (size_t i = 0; i < B; ++i) {
            tm_example &e = ex[i];
            if (e.cover >= P.K_COVER) continue;
            uint32_t p_req = (uint32_t)(((uint64_t)e.py_q32 * (uint64_t)P.rho_seed_q32) >> 32);
            if (!tm_bern(p_req, P.seed, TM_S_SEED, e.a, 0, 0)) continue;
            diag[TM_D_N_SEED_REQUESTS] += 1;
            if (free_heap.empty() || served[e.y] >= P.SEED_CAP) continue;
            served[e.y] += 1;                                        /* N4 */
            uint32_t lits[3]; uint32_t n = tm_seed_literals(&P, &X, &e, lits);
            if (n == 0) continue;
            uint32_t j = free_heap.top(); free_heap.pop();
            tm_seed_clause(&P, &S, &e, j, lits, n);
            changed.push_back(j);
            diag[TM_D_N_SEEDED] += 1;
        }
        /* refresh of changed clauses and index rebuild, ascending (oracle: sorted(changed)) */
        std::sort(changed.begin(), changed.end());
        changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
        for (uint32_t j : changed) refresh(j);
        if ((S.step + 1) % P.SWEEP_PERIOD == 0) sweep();               /* N1 */
    }

    uint32_t age(uint32_t since) const { return (uint32_t)((S.step & TM_U32_MAX) - since); }   /* 32-bit wrap */

    void recycle(uint32_t j) {
        tm_recycle(&P, &S, j);
        refresh(j);
        free_heap.push(j);
        diag[TM_D_N_RECYCLED] += 1;
    }

    void sweep() {   /* spec 5.7 (oracle _sweep) */
        uint32_t period = P.SWEEP_PERIOD;
        for (uint32_t j = 0; j < P.C; ++j) {
            if (!alive[j]) continue;
            tm_settle(&P, &S, j);
            refresh(j);
            if (anchor[j] == TM_ANCHOR_NONE || !tm_owns_any(&P, &S, j) || age(last_tp_step[j]) > P.DEAD_SWEEPS * period) recycle(j);
        }
        if (free_heap.size() < P.min_free) {
            std::vector<std::pair<uint32_t, uint32_t>> old;
            for (uint32_t j = 0; j < P.C; ++j) if (alive[j] && age(birth_step[j]) > 2 * period) old.emplace_back(stat_tp[j], j);
            std::sort(old.begin(), old.end());
            for (auto &pr : old) { if (free_heap.size() >= P.min_free) break; recycle(pr.second); }
        }
        for (uint32_t j = 0; j < P.C; ++j) { stat_fire[j] >>= 1; stat_tp[j] >>= 1; }
    }

    void train_dense(tm_example &ex, Fwd &fw) {   /* spec 6 (oracle _train_dense) */
        std::vector<uint32_t> scratch;
        forward(ex, fw, scratch, 1);
        diag[TM_D_N_CANDIDATES] += fw.n_candidates;
        diag[TM_D_N_FIRED] += fw.fired.size();
        std::vector<uint8_t> fires(P.C, 0);
        for (uint32_t j : fw.fired) fires[j] = 1;
        #pragma omp parallel
        {
            uint64_t d[TM_D_COUNT]; std::memset(d, 0, sizeof d);
            #pragma omp for schedule(static)
            for (uint32_t j = 0; j < P.C; ++j) tm_dense_clause_feedback(&P, &X, &S, &ex, j, fires[j], fw.scores.data(), d);
            #pragma omp critical
            for (int q = 0; q < TM_D_COUNT; ++q) diag[q] += d[q];
        }
        G[ex.y] += (uint64_t)(ex.py_q32 >> 16);
    }

    /* ---------------- prediction ---------------- */
    void predict(uint32_t epoch, const uint32_t *ids, size_t n, const uint8_t *bitmap, int32_t *out) {
        #pragma omp parallel
        {
            std::vector<uint32_t> scratch; Fwd fw; fw.scores.assign(P.V, 0);
            #pragma omp for schedule(dynamic, 16)
            for (size_t r = 0; r < n; ++r) {
                tm_example ex; tm_make_example(&P, &X, &ex, epoch, ids[r], bitmap);
                forward(ex, fw, scratch, 0);
                int32_t *row = out + r * P.V;
                for (uint32_t c = 0; c < P.V; ++c) row[c] = 0;
                for (uint32_t c : fw.touched) row[c] = fw.scores[c];
            }
        }
    }
    uint32_t fired(uint32_t epoch, uint32_t id, const uint8_t *bitmap, uint32_t *out, uint32_t cap, int bruteforce) {
        tm_example ex; tm_make_example(&P, &X, &ex, epoch, id, bitmap);
        std::vector<uint32_t> f;
        if (bruteforce) {
            for (uint32_t j = 0; j < P.C; ++j)
                if (is_dense ? tm_fires_dense(&P, &X, &S, &ex, j, 0) : tm_fires_live(&P, &X, &S, &ex, j)) f.push_back(j);
        } else {
            std::vector<uint32_t> scratch; Fwd fw; fw.scores.assign(P.V, 0);
            forward(ex, fw, scratch, 0);
            f = fw.fired;
        }
        uint32_t n = (uint32_t)std::min<size_t>(f.size(), cap);
        for (uint32_t i = 0; i < n; ++i) out[i] = f[i];
        return (uint32_t)f.size();
    }
};

} // namespace

extern "C" {

void *tm_create(const tm_params *P, const tm_corpus *X) {
    Engine *e = new Engine();
    e->P = *P; e->X = *X;
    e->is_dense = P->dense_equivalent != 0; e->is_tabular = P->mode == TM_MODE_TABULAR;
    e->alloc();
    if (e->is_dense) e->init_dense();
    e->rebuild_derived();
    std::memset(e->diag, 0, sizeof e->diag);
    return e;
}
void tm_destroy(void *h) { delete static_cast<Engine *>(h); }

/* state export: the caller provides buffers of the canonical sizes */
void tm_get_state(void *h, uint32_t *lit_id, uint8_t *lit_state, uint16_t *vote_cls, int16_t *vote_w,
                  uint64_t *decay_mark, uint64_t *own_fire_mass, uint32_t *stat_fire, uint32_t *stat_tp,
                  uint32_t *birth_step, uint32_t *last_tp_step, uint8_t *alive, uint64_t *G, uint64_t *step) {
    Engine *e = static_cast<Engine *>(h);
    std::memcpy(lit_id, e->lit_id.data(), e->lit_id.size() * sizeof(uint32_t));
    std::memcpy(lit_state, e->lit_state.data(), e->lit_state.size());
    std::memcpy(vote_cls, e->vote_cls.data(), e->vote_cls.size() * sizeof(uint16_t));
    std::memcpy(vote_w, e->vote_w.data(), e->vote_w.size() * sizeof(int16_t));
    std::memcpy(decay_mark, e->decay_mark.data(), e->decay_mark.size() * sizeof(uint64_t));
    std::memcpy(own_fire_mass, e->own_fire_mass.data(), e->own_fire_mass.size() * sizeof(uint64_t));
    std::memcpy(stat_fire, e->stat_fire.data(), e->stat_fire.size() * sizeof(uint32_t));
    std::memcpy(stat_tp, e->stat_tp.data(), e->stat_tp.size() * sizeof(uint32_t));
    std::memcpy(birth_step, e->birth_step.data(), e->birth_step.size() * sizeof(uint32_t));
    std::memcpy(last_tp_step, e->last_tp_step.data(), e->last_tp_step.size() * sizeof(uint32_t));
    std::memcpy(alive, e->alive.data(), e->alive.size());
    std::memcpy(G, e->G.data(), e->G.size() * sizeof(uint64_t));
    *step = e->S.step;
}
void tm_load_state(void *h, const uint32_t *lit_id, const uint8_t *lit_state, const uint16_t *vote_cls, const int16_t *vote_w,
                   const uint64_t *decay_mark, const uint64_t *own_fire_mass, const uint32_t *stat_fire, const uint32_t *stat_tp,
                   const uint32_t *birth_step, const uint32_t *last_tp_step, const uint8_t *alive, const uint64_t *G, uint64_t step) {
    Engine *e = static_cast<Engine *>(h);
    std::memcpy(e->lit_id.data(), lit_id, e->lit_id.size() * sizeof(uint32_t));
    std::memcpy(e->lit_state.data(), lit_state, e->lit_state.size());
    std::memcpy(e->vote_cls.data(), vote_cls, e->vote_cls.size() * sizeof(uint16_t));
    std::memcpy(e->vote_w.data(), vote_w, e->vote_w.size() * sizeof(int16_t));
    std::memcpy(e->decay_mark.data(), decay_mark, e->decay_mark.size() * sizeof(uint64_t));
    std::memcpy(e->own_fire_mass.data(), own_fire_mass, e->own_fire_mass.size() * sizeof(uint64_t));
    std::memcpy(e->stat_fire.data(), stat_fire, e->stat_fire.size() * sizeof(uint32_t));
    std::memcpy(e->stat_tp.data(), stat_tp, e->stat_tp.size() * sizeof(uint32_t));
    std::memcpy(e->birth_step.data(), birth_step, e->birth_step.size() * sizeof(uint32_t));
    std::memcpy(e->last_tp_step.data(), last_tp_step, e->last_tp_step.size() * sizeof(uint32_t));
    std::memcpy(e->alive.data(), alive, e->alive.size());
    std::memcpy(e->G.data(), G, e->G.size() * sizeof(uint64_t));
    e->S.step = step;
    e->rebuild_derived();
}

/* train_batch takes the batch size explicitly; the permutation is the engine's own tm_perm. */
int tm_train_batch_b(void *h, uint32_t epoch, uint32_t batch_index, uint32_t batch, uint64_t *diag_out) {
    Engine *e = static_cast<Engine *>(h);
    uint64_t n = e->is_tabular ? e->X.n_rows : e->X.n_targets;
    uint64_t lo = (uint64_t)batch_index * batch;
    if (lo >= n) return -1;
    uint64_t hi = std::min<uint64_t>(n, lo + batch);
    std::vector<uint32_t> ids;
    ids.reserve(hi - lo);
    for (uint64_t x = lo; x < hi; ++x) {
        uint64_t idx = tm_perm(e->P.seed, epoch, x, n);
        ids.push_back(e->is_tabular ? (uint32_t)idx : e->X.targets[idx]);
    }
    e->train(epoch, ids);
    std::memcpy(diag_out, e->diag, sizeof e->diag);
    return 0;
}
int tm_train_explicit(void *h, uint32_t epoch, const uint32_t *ids, uint32_t n, uint64_t *diag_out) {
    Engine *e = static_cast<Engine *>(h);
    std::vector<uint32_t> v(ids, ids + n);
    e->train(epoch, v);
    std::memcpy(diag_out, e->diag, sizeof e->diag);
    return 0;
}
int tm_predict_scores(void *h, uint32_t epoch, const uint32_t *ids, uint32_t n, const uint8_t *bitmap, int32_t *out) {
    static_cast<Engine *>(h)->predict(epoch, ids, n, bitmap, out);
    return 0;
}
uint32_t tm_fired_clauses(void *h, uint32_t epoch, uint32_t id, const uint8_t *bitmap, uint32_t *out, uint32_t cap, int bruteforce) {
    return static_cast<Engine *>(h)->fired(epoch, id, bitmap, out, cap, bruteforce);
}
uint32_t tm_n_free(void *h) { return (uint32_t)static_cast<Engine *>(h)->free_heap.size(); }
int tm_omp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // extern "C"
